# Using public/private key to access a private GitHub repo


## Create a public/private key pair on the system to be granted access

```
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Don't choose a passphrase.  Make sure the resulting files were stored in `~/.ssh`.
I will assume the private and public pair has the name `xxx` and `xxx.pub`.


## Grant github repo access to the key

+ Click on your user Github profile picture (top right) and select 'Settings'.
+ Select SSH and GPG keys
+ In SSH keys, click 'New SSH Key'
+ Enter a title, and copy `xxx.pub` key to the 'Key' field.

## Install the key pair to to the remote machine

Copy the key pair (xxx and xxx.pub) to the remote machine's `~/.ssh` directory.

Make sure the private file has restricted permissions:
```
chmod 600 ~/.ssh/xxx
```

## Set GitHub identity on remote system

From:

https://superuser.com/questions/232373/how-to-tell-git-which-private-key-to-use

If necessary, create the file `~/.ssh/config` and include this text:

```
Host github.com
 HostName github.com
 IdentityFile ~/.ssh/xxx
```

Make sure it has restricted permissions:
```
chmod 600 ~/.ssh/config
```

## Clone a repo to the remote machine

Create a directory (outside of any existing git repos):

```
mkdir ~/experiment
cd ~/experiment

git clone git@github.com:jpsember/java-ml.git
```


## Alternative

(From `https://superuser.com/questions/232373/`)

Avoids having to create or modify `~/.ssh/config`.


Create an environment variable:

```
export GIT_SSH_COMMAND="ssh -i ~/.ssh/id_github_alt -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"
```

Then git commands will work, e.g.:

```
git pull
```
