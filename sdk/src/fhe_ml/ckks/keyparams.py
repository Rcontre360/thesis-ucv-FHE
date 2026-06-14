import base64
import json
from dataclasses import dataclass

from fhe_ml.backend import CKKSGaloiskey, CKKSRelinkey


@dataclass
class KeyParams:
    relin_key: CKKSRelinkey
    galois_key: CKKSGaloiskey | None

    def serialize(self) -> str:
        galois = self.galois_key.to_bytes() if self.galois_key is not None else None
        return json.dumps(
            {
                "relin": base64.b64encode(self.relin_key.to_bytes()).decode("ascii"),
                "galois": None
                if galois is None
                else base64.b64encode(galois).decode("ascii"),
            }
        )

    @classmethod
    def deserialize(cls, blob: str) -> "KeyParams":
        d = json.loads(blob)
        relin = CKKSRelinkey.from_bytes(base64.b64decode(d["relin"]))
        galois = (
            None
            if d["galois"] is None
            else CKKSGaloiskey.from_bytes(base64.b64decode(d["galois"]))
        )
        return cls(relin_key=relin, galois_key=galois)
