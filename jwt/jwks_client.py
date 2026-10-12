from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from functools import lru_cache
from ssl import SSLContext
from typing import Any
from urllib.error import HTTPError, URLError

from .api_jwk import PyJWK, PyJWKSet
from .api_jwt import decode_complete as decode_token
from .exceptions import PyJWKClientConnectionError, PyJWKClientError
from .jwk_set_cache import JWKSetCache


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class PyJWKClient:
    def __init__(
        self,
        uri: str,
        cache_keys: bool = False,
        max_cached_keys: int = 16,
        cache_jwk_set: bool = True,
        lifespan: float = 300,
        headers: dict[str, Any] | None = None,
        timeout: float = 30,
        ssl_context: SSLContext | None = None,
        cooldown_duration: float = 30,
    ):
        if headers is None:
            headers = {}
        self.uri = uri
        self.jwk_set_cache: JWKSetCache | None = None
        self.headers = headers
        self.timeout = timeout
        self.ssl_context = ssl_context
        # Unknown key IDs can trigger one forced refresh after the cooldown
        # period configured by ``cooldown_duration``. Every successful fetch
        # starts this cooldown, including the initial fetch and cache-expiry
        # fetches. Set ``cooldown_duration`` to ``0`` to disable this behavior.
        # The cooldown is bypassed when ``cache_jwk_set`` is ``False``.
        if cooldown_duration < 0:
            raise PyJWKClientError(
                "Cooldown duration must be greater than or equal to 0, "
                f'the input is "{cooldown_duration}"'
            )
        if not math.isfinite(cooldown_duration):
            raise PyJWKClientError(
                f'Cooldown duration must be finite, the input is "{cooldown_duration}"'
            )
        self.cooldown_duration = cooldown_duration
        self._last_successful_fetch: float | None = None
        self._client_lock = threading.RLock()

        if cache_jwk_set:
            # Init jwt set cache with default or given lifespan.
            # Default lifespan is 300 seconds (5 minutes).
            if lifespan <= 0:
                raise PyJWKClientError(
                    f'Lifespan must be greater than 0, the input is "{lifespan}"'
                )
            self.jwk_set_cache = JWKSetCache(lifespan)
        else:
            self.jwk_set_cache = None

        if cache_keys:
            # Cache signing keys
            get_signing_key = lru_cache(maxsize=max_cached_keys)(self.get_signing_key)
            # Ignore mypy (https://github.com/python/mypy/issues/2427)
            self.get_signing_key = get_signing_key  # type: ignore[method-assign]

    def fetch_data(self) -> Any:
        jwk_set: Any = None
        try:
            r = urllib.request.Request(url=self.uri, headers=self.headers)
            handlers: list[Any] = [_NoRedirectHandler()]
            if self.ssl_context is not None:
                handlers.append(urllib.request.HTTPSHandler(context=self.ssl_context))
            opener = urllib.request.build_opener(*handlers)
            with opener.open(r, timeout=self.timeout) as response:
                jwk_set = json.load(response)
        except (URLError, TimeoutError) as e:
            if isinstance(e, HTTPError):
                e.close()
            raise PyJWKClientConnectionError(
                f'Fail to fetch data from the url, err: "{e}"'
            ) from e
        else:
            self._last_successful_fetch = time.monotonic()
            return jwk_set
        finally:
            if self.jwk_set_cache is not None:
                self.jwk_set_cache.put(jwk_set)

    def get_jwk_set(self, refresh: bool = False) -> PyJWKSet:
        data = None
        if self.jwk_set_cache is not None and not refresh:
            data = self.jwk_set_cache.get()

        if data is None:
            data = self.fetch_data()

        if not isinstance(data, dict):
            raise PyJWKClientError("The JWKS endpoint did not return a JSON object")

        return PyJWKSet.from_dict(data)

    def get_signing_keys(self, refresh: bool = False) -> list[PyJWK]:
        jwk_set = self.get_jwk_set(refresh)
        return self._get_signing_keys_from_jwk_set(jwk_set)

    @staticmethod
    def _get_signing_keys_from_jwk_set(jwk_set: PyJWKSet) -> list[PyJWK]:
        signing_keys = [
            jwk_set_key
            for jwk_set_key in jwk_set.keys
            if jwk_set_key.public_key_use in ["sig", None] and jwk_set_key.key_id
        ]

        if not signing_keys:
            raise PyJWKClientError("The JWKS endpoint did not contain any signing keys")

        return signing_keys

    def get_signing_key(self, kid: str) -> PyJWK:
        with self._client_lock:
            signing_keys = self.get_signing_keys()
            signing_key = self.match_kid(signing_keys, kid)

            if not signing_key:
                # If no matching signing key from the jwk set, refresh the jwk set
                # and try again, unless a fetch happened within the cooldown period.
                cooling_down = (
                    self.jwk_set_cache is not None
                    and self._last_successful_fetch is not None
                    and time.monotonic() - self._last_successful_fetch
                    < self.cooldown_duration
                )
                if not cooling_down:
                    signing_keys = self.get_signing_keys(refresh=True)
                    self._last_successful_fetch = time.monotonic()
                    signing_key = self.match_kid(signing_keys, kid)

                if not signing_key:
                    raise PyJWKClientError(
                        f'Unable to find a signing key that matches: "{kid}"'
                    )

            return signing_key

    def get_signing_key_from_jwt(self, token: str | bytes) -> PyJWK:
        unverified = decode_token(token, options={"verify_signature": False})
        header = unverified["header"]
        return self.get_signing_key(header.get("kid"))

    @staticmethod
    def match_kid(signing_keys: list[PyJWK], kid: str) -> PyJWK | None:
        signing_key = None

        for key in signing_keys:
            if key.key_id == kid:
                signing_key = key
                break

        return signing_key
