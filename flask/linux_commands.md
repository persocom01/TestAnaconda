<!-- path_to_file normally begins with ~/, which means the file will be copied
to the home directory instead of root, which is normally prohibited. -->
<!-- To copy file from remote to local, reverse the positions of file_path and
user_name@aws_public_dns:path_to_file -->
scp -i key_path file_path user_name@aws_public_dns:path_to_file

<!-- Copy folder to aws. Not the -r. It copies the folder itself, and not just
folder contents. -->
scp -i path/to/key -r directory/to/copy user@ec2-xx-xx-xxx-xxx.compute-1.amazonaws.com:path/to/directory

<!-- Install git so the repo can be easily cloned onto the server. -->
sudo apt-get install git
git clone https://link_to_repo target_dir
