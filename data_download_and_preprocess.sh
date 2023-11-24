#Download Duke dataset
#wget http://www.duke.edu/~sf59/Datasets/2015_BOE_Chiu2.zip

#Download UMN dataset
#wget http://people.ece.umn.edu/users/parhi/.DATA/OCT/DME/UMN_Method_DukeDataset.mat

#Unzip Downloaded File
#unzip 2015_BOE_Chiu2.zip

#Prepare and create the required data directories
#mkdir -p DukeData
#mkdir -p DukeData/train DukeData/val DukeData/test

#mkdir -p UMNData2
#mkdir -p UMNData2/train UMNData2/val UMNData2/test

#Run preprocessing code
python preprocessing.py "./2015_BOE_Chiu" "DukeData"
python preprocessing.py --dataset "UMN" "UMN_Method_DukeDataset.mat" "UMNData2"
