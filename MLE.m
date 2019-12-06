%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Problem A %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
load('TrainingSamplesDCT_8_new.mat');

D_BG= TrainsampleDCT_BG;
D_FG= TrainsampleDCT_FG;
BG_cor = my_cor(D_BG);
FG_cor = my_cor(D_FG);
BG_mu = my_mean(D_BG);
FG_mu = my_mean(D_FG);

% Sample number of BG and FG
count_BG = size(D_BG,1);
count_FG = size(D_FG,1);

% ML estimates for the class priors
P_Y=[count_BG  count_FG]/(count_FG+count_BG);

img = imread('cheetah.bmp');
imshow(img);
% Normalize the image and extract the feature
img = double(img)/255;
% img2dct is used to compute the dct zigzaged feature
[I,features,rol,col] = img2dct(img);
disp("feature completed")

brol = rol-7;
bcol = col-7;
dsize = size(features,1);
mask = zeros(brol,bcol);
h=waitbar(0,'Classifying');
for b=0:dsize-1
    j = mod(b,bcol);
    i =(b-j)/(bcol);
    str=['Classifying...',num2str(100*b/dsize),'%'];
    waitbar(b/dsize,h,str);
    if MVNB(FG_cor,FG_mu,features(b+1,:),P_Y(2))<MVNB(BG_cor,BG_mu,features(b+1,:),P_Y(1))
        mask(i+1,j+1)=255;
    end
end
delete(h);
mask_name='cheetah_premask_MLE.png';
imwrite(mask,mask_name);


function mean = my_mean(data)
    mean = sum(data)/size(data,1);
end
function cov = my_cor(data)
    % Compute covariance matrix
    % Input
    % data: input data with M rol(M samples) and N column(N feature)
    % Output
    % cov: covariance matrix
    mean = sum(data)/size(data,1);
    cov = (data-mean)'*(data-mean)/size(data,1);
end

function posterior = MVNB(C,mu,x,P_y)
    % Compute the postior using BDR
    % Input
    % C: covariance matrix
    % mu: The means of the data
    % x : The input sample
    % P_y: prior
    % Output
    % posterior: posterior probability given x
    posterior = (x-mu)*inv(C)*(x-mu).'+log(det(C))-2*log(P_y);
end

function [dctimg,dctvectors,rol,col]=img2dct(img)
    % Transform the original image to DCT features
    % Input
    % img:original Image
    % Output
    % dctimg: A combination of all DCT blocks without flattening(For visualizing purpose)
    % dctvectors: all DCT feature vectors after flattening
    % rol: the row number after padding 
    % col: the column number after padding 
    
    % read in the zigzag order file and transfer to order index
    filezz = fopen('J:\Learning\UCSD\classes\271A\ece271_hw1\Zig-Zag Pattern.txt','r');
    zigzag = fscanf(filezz,'%f',[8 8]);
    fclose(filezz);
    index = zigzag2index(zigzag);
    % image padding
    img=padarray(img,[7 7],'symmetric','post');
    [rol,col,channel] = size(img);
    stride = 1;
    dctimg=zeros((rol-7)*8,(col-7)*8,channel);
    vsize = (rol-7)*(col-7);
    dctvectors=zeros(vsize,64);
    for i=1:stride:rol-7
        for j=1:stride:col-7
        % for each block, compute its dct
            block = dct2(img(i:i+7,j:j+7));
            feature_index=(i-1)*(col-7)+j;
            % transform it into zigzag order
                for m = 1:64
                    id = index(m,:);
                    dctvectors(feature_index,m)=block(id(1),id(2));
                end
            dctimg(8*i:8*i+7,8*j:8*j+7) = block;
        end
    end
% save('cheetah.mat','dctvectors');
end

function index = zigzag2index(zigzag)
    % Transfer zigzag value to zigzag order
    % Input
    % zigzag: the Zigzag matrix comes from the txt file
    % Output
    % index: a vetor that its i^th value reprents the index in the 2D 8*8 matrix
    index = zeros(64,2);
    for i=0:63
        [x y] = find(zigzag==i);
        index(i+1,1)=x;
        index(i+1,2)=y;
    end
end