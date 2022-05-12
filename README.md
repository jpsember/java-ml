Jeff's Java classes for machine learning


# Configuring machine for PyTorch

To install on a Linux machine that is GPU enabled:




```
sudo apt install libjpeg-dev zlib1g-dev
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```



Derived from https://pytorch.org/get-started/locally/
And https://stackoverflow.com/questions/44043906


Earlier attempts:



```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

(These instructions were from https://pytorch.org/get-started/locally/ )


Trying this on Ubuntu resulted in:

```

      File "/home/.pyenv/versions/3.7.3/lib/python3.7/site-packages/setuptools/command/build_ext.py", line 79, in run
        _build_ext.run(self)
      File "/home/.pyenv/versions/3.7.3/lib/python3.7/site-packages/setuptools/_distutils/command/build_ext.py", line 339, in run
        self.build_extensions()
      File "/tmp/pip-install-astiq52e/pillow/setup.py", line 804, in build_extensions
        raise RequiredDependencyException(f)
    __main__.RequiredDependencyException: jpeg

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-install-astiq52e/pillow/setup.py", line 1009, in <module>
        raise RequiredDependencyException(msg)
    __main__.RequiredDependencyException:

    The headers or library files could not be found for jpeg,
    a required dependency when compiling Pillow from source.

    Please see the install instructions at:
       https://pillow.readthedocs.io/en/latest/installation.html



    ----------------------------------------
Command "/home/.pyenv/versions/3.7.3/bin/python3.7 -u -c "import setuptools, tokenize;__file__='/tmp/pip-install-astiq52e/pillow/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))" install --record /tmp/pip-record-zs96gxot/install-record.txt --single-version-externally-managed --compile" failed with error code 1 in /tmp/pip-install-astiq52e/pillow/
```
