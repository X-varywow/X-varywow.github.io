

## _hashlib_


```python
import hashlib

def hash_md5(password):
    encoded_string = password.encode()
    hexdigest = hashlib.md5(encoded_string).hexdigest()
    return hexdigest
```




</br>

## _jwt_

https://pyjwt.readthedocs.io/en/stable/

JWT JSON Web Tokens, HS256 是 HMAC 和 SHA-256 的组合，JWT 最常用的加密算法之一，确保 TOKEN 的完整性，即验证数据部分没有被篡改

```python
import jwt
encoded_jwt = jwt.encode({"some": "payload"}, "secret", algorithm="HS256")
print(encoded_jwt)
# -> eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb21lIjoicGF5bG9hZCJ9.4twFt5NiznN84AWoo1d7KO1T_yoc0Z6XOpOVswacPZg

jwt.decode(encoded_jwt, "secret", algorithms=["HS256"])
# -> {'some': 'payload'}
```