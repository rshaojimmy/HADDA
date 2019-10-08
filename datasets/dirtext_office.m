clc;clear;

rootdir = 'E:\research3\datasets_info\domain_adaptation_datasets\office\webcam\';
resultsDir = 'E:\research3\datasets_info\domain_adaptation_datasets\office\webcam\image_list.txt';

rowall = {};

sbjlist = dir([rootdir]);
sbjlist = sbjlist(~ismember({sbjlist.name},{'.','..'}));

for sbjNum = 1 : length(sbjlist)
    sbjname = sbjlist(sbjNum).name;
    
    imgdir = [rootdir, sbjname '\'];
    
    imglist = dir([imgdir,'*.jpg']);

    for imgNum = 1 : length(imglist)  

        imgname = imglist(imgNum).name;
        
        rowone = [sbjname '/' imgname ' ' num2str(sbjNum-1)];
        rowall = [rowall; rowone];
   
    end    
end

fid=fopen(resultsDir,'wt'); 
[m,n]=size(rowall); 
for i=1:1:m 
    fprintf(fid,'%s\n',rowall{i,:}); 
end 
fclose(fid); 




