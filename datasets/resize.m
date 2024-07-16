%matlab 从一个文件夹中批量读取、处理并保存图片文件到另外一个文件夹
%测试平台：windows 10, matlab 2010R 64b
%编写日期：20180717
%zhouxianen
 
clear;
clc;
close all;
 
srcFace = 'C:\Users\lzhan\Desktop\sisr\SISR\BSRN\BSRN\datasets\Set5\GTmod4';%被读取文件的存放目录（根据自己需要更改设置）
fileSavePath='C:\Users\lzhan\Desktop\sisr\SISR\BSRN\BSRN\datasets\Set5\re';%文件保存目录（根据自己需要更改设置）
src=srcFace;
srcsuffix='.png';%被读取的文件名后缀（根据被读取文件的实际文件类型设置）
srcsuffixSave='.png';%保存文件名后缀（根据自己需要更改设置）
files = dir(fullfile(src, strcat('*', srcsuffix)));
doDispOrSave = true;% 是否显示或保存图像；可以设置为：true 或者 false
for file_i= 1 : length(files)
    disp(file_i);%显示当前处理的文件序号
    srcName = files(file_i).name;
    noSuffixName = srcName(1:end-4);
    srcName1=files(file_i).name;
    pathImgName=sprintf('%s%s%s',src,'\',srcName1);
    imgSrc=imread(pathImgName);%读入图像
    %对读入的图像进行尺度缩放处理
    imgResize=imresize(imgSrc,0.5);
    %显示或者保存图像
    savePathName=sprintf('%s%s%s%s',fileSavePath,'\',noSuffixName,srcsuffixSave);
    imwrite(imgResize,savePathName);
end
