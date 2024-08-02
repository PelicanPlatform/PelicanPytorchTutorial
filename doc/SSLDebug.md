# Debug diary：unable to get local issuer certificate

## **Problem Background**

Today when I try to run my python script on HTC's executed point, (submitting by HTCondor), I got following error when I using `fs.get()` (The fs here is a PelicanFileSystem implement in pelicanfs, it can also be some file system implement in fsspec)

```
File "/var/lib/condor/execute/slot1/dir_1842404/myenv/lib/python3.10/ssl.py", line 975, in do_handshake
    self._sslobj.do_handshake()
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)
File "/var/lib/condor/execute/slot1/dir_1842404/myenv/lib/python3.10/site-packages/aiohttp/connector.py", line 1027, in _wrap_create_connection
    raise ClientConnectorCertificateError(req.connection_key, exc) from exc
aiohttp.client_exceptions.ClientConnectorCertificateError: Cannot connect to host osg-htc.org:443 ssl:True [SSLCertVerificationError: (1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)')]
```

## [**What is an SSL certificate?**](https://www.cloudflare.com/learning/ssl/what-is-ssl/)

SSL can only be implemented by websites that have an [SSL certificate](https://www.cloudflare.com/learning/ssl/what-is-an-ssl-certificate/) (technically a "TLS certificate"). An SSL certificate is like an ID card or a badge that proves someone is who they say they are. SSL certificates are stored and displayed on the Web by a website's or application's server.

One of the most important pieces of information in an SSL certificate is the website's public [key](https://www.cloudflare.com/learning/ssl/what-is-a-cryptographic-key/). The [public key](https://www.cloudflare.com/learning/ssl/how-does-public-key-encryption-work/) makes encryption and authentication possible. A user's device views the public key and uses it to establish secure encryption keys with the web server. Meanwhile the web server also has a private key that is kept secret; the private key decrypts data encrypted with the public key.

Certificate authorities (CA) are responsible for issuing SSL certificates.

## **Some existed fix**

I read a lot of post on stack over flow, but these method, like `pip install certifi`, is what I have done.

These didn't solve my problem, but may solve yours, so here are some posts I read for your reference:

- [Unable to get local issuer certificate when using requests](https://stackoverflow.com/questions/51925384/unable-to-get-local-issuer-certificate-when-using-requests)
- ["SSL: certificate_verify_failed" error when scraping](https://stackoverflow.com/questions/34503206/ssl-certificate-verify-failed-error-when-scraping-https-www-thenewboston-co)
- [Resolving SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)’))) Ensuring Secure API Connections in Python](https://medium.com/@vkmauryavk/resolving-sslcertverificationerror-certificate-verify-failed-unable-to-get-local-issuer-515d7317454f)

## **My Fix**

when I run

```python
import certifi
print(certifi.where())
```

It do give me a path, it should be something like

```
<Your rootDir>/myenv/lib/python3.10/site-packages/certifi/cacert.pem
```

And when I do `conda config --show ssl_verify` , it shows `ssl_verify: True`

Do `conda list certifi`

It gives me

```
# packages in environment at /home/hzhao292/miniconda3:
#
# Name                    Version                   Build  Channel
ca-certificates           2024.7.4             hbcca054_0    conda-forge
certifi                   2024.7.4           pyhd8ed1ab_0    conda-forge
```

These three verification results all tells me, I do have valid SSL certification. And I also know its path right now.

However, when I run:

```
python -c "import ssl; print(ssl.get_default_verify_paths())"
```

I get

```
DefaultVerifyPaths(cafile=None, capath=None, openssl_cafile_env='SSL_CERT_FILE', openssl_cafile='$ENVDIR/ssl/cert.pem', openssl_capath_env='SSL_CERT_DIR', openssl_capath='$ENVDIR/ssl/certs')
```

Therefore, the problem comes to light. I do have certificates installed, however due to some kind of reason, when I running my script, python didn't find it.

When you met this problem when using `request()`, you may add verify path to the parameter, like this:

```
requests.post(url, params, verify='/path/to/certifi.pem')
```

You can also set `verify=False`, to avoid verify your SSL, however, this is not recommended, especially in production, because it's not secure. Although it maybe okay in test enviornment, it will foster an insecurity culture. We should find the root of the problem rather than just bypass it.

However, becasue I am using the get method from custom file system, which inherit a lot. I tried to pass the verify parameter, but it doesn't work. Therefore, I am wondering whether I can tell python to find the path globally.

Therefore, I add the path of certificate manually by shell.

```bash
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export SSL_CERT_DIR=$(dirname $(python -c "import certifi; print(certifi.where())"))
```

Then, run the same python script before, `python -c "import ssl; print(ssl.get_default_verify_paths())"`, we get

```
DefaultVerifyPaths(cafile='/var/lib/condor/execute/slot1/dir_1107337/myenv/lib/python3.10/site-packages/certifi/cacert.pem', capath='/var/lib/condor/execute/slot1/dir_1107337/myenv/lib/python3.10/site-packages/certifi', openssl_cafile_env='SSL_CERT_FILE', openssl_cafile='$ENVDIR/ssl/cert.pem', openssl_capath_env='SSL_CERT_DIR', openssl_capath='$ENVDIR/ssl/certs')
```

Which indicats python can find the path right now! Uh huh!

Then my get request works! Yeah.
