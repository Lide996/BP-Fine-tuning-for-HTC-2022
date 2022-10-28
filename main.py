import funcs
import os
import argparse


parser = argparse.ArgumentParser(description='Limited angle computed tomography')

parser.add_argument('--data_dir', default='./data/',type=str, help='Folder where the input files are located')
parser.add_argument('--out_dir', default='./output/', type=str, help=' Folder where the output files is stored')
parser.add_argument('--group_number', default=1, type=int, help=' Group category number')
args = parser.parse_args()


if __name__ == "__main__":
    
    group_number=args.group_number
    data_folder=args.data_dir
    output_folder=args.out_dir
    
    ##list all files in data folder
    data_list=os.listdir(data_folder)
    ##find all mat files in the list
    data_list=funcs.find_mat(data_list)

    ##check whether the list is empty
    if not data_list:
       raise ValueError('There is no mat file in given folder path.') 
    ##reconstruct the phantoms
    for i in range(len(data_list)):
        ##get data path
        load_file_name=data_list[i]
        print('processing data:',load_file_name)

        output_file_name=load_file_name[0:-4]+'.png'
        data_path=data_folder+load_file_name
        output_path=output_folder+output_file_name
        ##reconstruct
        funcs.Load_process(data_path,output_path,group_number)        
    

