%matlab ��һ���ļ�����������ȡ����������ͼƬ�ļ�������һ���ļ���
%����ƽ̨��windows 10, matlab 2010R 64b
%��д���ڣ�20180717
%zhouxianen
 
clear;
clc;
close all;
 
srcFace = 'C:\Users\lzhan\Desktop\sisr\SISR\BSRN\BSRN\datasets\Set5\GTmod4';%����ȡ�ļ��Ĵ��Ŀ¼�������Լ���Ҫ�������ã�
fileSavePath='C:\Users\lzhan\Desktop\sisr\SISR\BSRN\BSRN\datasets\Set5\re';%�ļ�����Ŀ¼�������Լ���Ҫ�������ã�
src=srcFace;
srcsuffix='.png';%����ȡ���ļ�����׺�����ݱ���ȡ�ļ���ʵ���ļ��������ã�
srcsuffixSave='.png';%�����ļ�����׺�������Լ���Ҫ�������ã�
files = dir(fullfile(src, strcat('*', srcsuffix)));
doDispOrSave = true;% �Ƿ���ʾ�򱣴�ͼ�񣻿�������Ϊ��true ���� false
for file_i= 1 : length(files)
    disp(file_i);%��ʾ��ǰ������ļ����
    srcName = files(file_i).name;
    noSuffixName = srcName(1:end-4);
    srcName1=files(file_i).name;
    pathImgName=sprintf('%s%s%s',src,'\',srcName1);
    imgSrc=imread(pathImgName);%����ͼ��
    %�Զ����ͼ����г߶����Ŵ���
    imgResize=imresize(imgSrc,0.5);
    %��ʾ���߱���ͼ��
    savePathName=sprintf('%s%s%s%s',fileSavePath,'\',noSuffixName,srcsuffixSave);
    imwrite(imgResize,savePathName);
end
