"""Regression guard for config key naming collisions.

The ``custom_llm_*`` LLM provider keys share a ``custom_`` prefix with the
material ``custom_*`` keys. A prefix collision here could silently route one
provider's settings into another's env/keyring lookup (or a WebUI fallback),
so keep the two families distinct and cover it with a test.
"""

from app.config import config
from app.models.llm_provider import LLM_PROVIDER_REGISTRY

_LLM_SUFFIXES = ("api_key", "base_url", "model_name")


class TestConfigKeyNamingCollisions:
    def test_no_key_is_a_prefix_of_another_key(self):
        """No config key may be a prefix of another (``key + "_"``) key.

        Keys are underscore-separated, so ``custom_api_key`` being a prefix of
        ``custom_llm_api_key`` would be a collision: config lookups keyed on
        the shorter name would accidentally match the longer one.
        """
        keys = {key for (_section, key) in config._SECTION_KEY_TO_ENV}
        assert keys, "expected at least one documented config key"

        secret_keys = {key for key in keys if config._is_secret_key(key)}
        assert secret_keys, "expected at least one secret-sourced config key"

        for key in sorted(keys):
            for other in sorted(keys):
                if key == other:
                    continue
                assert not other.startswith(f"{key}_"), (
                    f"config key collision: {key!r} is a prefix of {other!r} "
                    f"(secret-sourced keys are scanned too)"
                )

    def test_provider_config_keys_do_not_collide_across_owners(self):
        """Each LLM provider's ``config_key()`` must own its entries exclusively.

        If ``provider.config_key("api_key")`` matched a ``_SECTION_KEY_TO_ENV``
        entry owned by a different provider/section (e.g. a material custom API
        key), the provider would silently read the wrong credential.
        """
        owners: dict[tuple[str, str], str] = {}
        for section, key in config._SECTION_KEY_TO_ENV:
            owner = section
            for provider in LLM_PROVIDER_REGISTRY:
                claimed = {
                    provider.config_key(suffix) for suffix in _LLM_SUFFIXES
                } | {
                    provider.config_key(field.config_suffix)
                    for field in provider.extra_fields
                }
                if key in claimed:
                    owner = provider.provider_id
                    break
            owners[(section, key)] = owner

        for provider in LLM_PROVIDER_REGISTRY:
            key = provider.config_key("api_key")
            for (section, existing_key), owner in owners.items():
                if existing_key != key:
                    continue
                assert owner == provider.provider_id, (
                    f"provider {provider.provider_id!r} config key {key!r} "
                    f"collides with {section}.{existing_key} owned by {owner!r}"
                )
