from fhe_sdk._backend import SecurityLevel as BackendSecurityLevel
from fhe_sdk._backend import KeyswitchingType as BackendKeyswitchingType

class SecurityLevel:
    SEC128 = BackendSecurityLevel.SEC128
    SEC192 = BackendSecurityLevel.SEC192
    SEC256 = BackendSecurityLevel.SEC256

class KeyswitchingType:
    METHOD_I  = BackendKeyswitchingType.METHOD_I
    METHOD_II = BackendKeyswitchingType.METHOD_II
