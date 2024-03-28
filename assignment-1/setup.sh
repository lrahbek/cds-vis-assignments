# create virtual enviorment
python -m venv env
# activate enviorment
source ./env/bin/activate
# install requirements
pip install --upgrade pip
sudo apt-get update
sudo apt-get install -y python3-opencv
pip install -r requirements.txt
# close the environment
deactivate