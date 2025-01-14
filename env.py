import subprocess

subprocess.run("conda create --name flower python=3.9.0 -y", shell=True)
subprocess.run("conda activate flower", shell=True)

subprocess.run("conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y", shell=True)

subprocess.run("pip install flwr==1.4.0", shell=True)
subprocess.run("pip install ray==1.11.1", shell=True)
subprocess.run("pip install segmentation-models-pytorch==0.3.4", shell=True)


subprocess.run("pip install matplotlib==3.9.4", shell=True)

subprocess.run("pip install opencv-contrib-python==4.10.0.84", shell=True)
subprocess.run("pip install pandas==2.2.3", shell=True)
subprocess.run("pip install Jinja2==3.1.5", shell=True)
subprocess.run("pip install albumentations==1.3.0", shell=True)

subprocess.run("pip install gdown==5.2.0", shell=True)
