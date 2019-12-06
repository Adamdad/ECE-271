%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Expectation Maximization %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('TrainingSamplesDCT_8_new.mat');

D_BG= TrainsampleDCT_BG;
D_FG= TrainsampleDCT_FG;

% Sample number of BG and FG
count_BG = size(D_BG,1);
count_FG = size(D_FG,1);
% ML estimates for the class priors
P_Y=[count_BG  count_FG]/(count_FG+count_BG);

% Number of components
C = 8;

% List of dimensions
dims = [1,2,4,8,16,24,32,40,48,56,64];

% Max iteration times
MAX_LOOP =200;

% number of mixture
mix_num=5;
arg_BG=[struct struct struct struct struct];
arg_FG =[struct struct struct struct struct];
disp(["Start training"]);

% Prepare for the image data
img = imread('cheetah.bmp');
% Normalize the image and extract the feature
img = double(img)/255;
% img2dct is used to compute the dct zigzaged feature
[I,features,rol,col] = img2dct(img);

t_D_BG= D_BG;
t_D_FG= D_FG;
dim = 64;
    
for mix=1:mix_num
        disp(['Dim = ',int2str(dim),'   Mix= ',int2str(mix)]);
        
        %%%%%%% Initialization %%%%%%%%
        % initlization mean
        [idx_BG,BG_mu]=kmeans( t_D_BG,C);
        BG_mu = BG_mu+randn(C,dim);
        [idx_FG,FG_mu]=kmeans(t_D_FG,C);
        FG_mu = FG_mu+randn(C,dim);

        % initlization prior
        pi_BG = rand(1,C);
        pi_BG = pi_BG/sum(pi_BG);

        pi_FG = rand(1,C);
        pi_FG = pi_FG/sum(pi_FG);
        % initlization convariance
        conv_BG = zeros(C,dim,dim);

        for i=1:C
            sigma =rand(1,dim);
            conv_BG(i,:,:)=diag(sigma+1);
             
        end
         conv_FG = zeros(C,dim,dim);

        for i=1:C
           sigma =rand(1,dim);
           conv_FG(i,:,:)=diag(sigma+1);
        end
        
        % Start EM algorithm
        for step=1:MAX_LOOP
            % E-step
            BG_hij = zeros(C,count_BG);
            for i=1:count_BG
                x = t_D_BG(i,:);
                h = zeros(C,1);
                for class=1:C
                    t_conv = reshape(conv_BG(class,:,:),[dim,dim]);
                    prob = mvnpdf(x,BG_mu(class,:),t_conv); 
                    h(class) =  prob*pi_BG(class);
                end
                h = h/sum(h); 
                BG_hij(:,i) = h;
           end

            FG_hij = zeros(C,count_FG);
            for i=1:count_FG
                x = t_D_FG(i,:);
                h = zeros(C,1);
                for class=1:C
                    t_conv = reshape(conv_FG(class,:,:),[dim,dim]);
                    prob = mvnpdf(x,FG_mu(class,:),t_conv);
                    h(class) = prob *pi_FG(class);
                end
                h = h/sum(h);
                FG_hij(:,i) = h;
            end
           
            % M-step
             
             
           for i=1:C
                hij_sum =  sum(BG_hij(i,:));
                BG_mu(i,:) =  BG_hij(i,:)* t_D_BG ./ hij_sum;
                sigma =  BG_hij(i,:)*(t_D_BG-BG_mu(i,:)).^2  ./ hij_sum;
                % In case the convariance diagnal elements are zeros
                sigma(find(sigma<1e-4))=1e-4;
                conv_BG(i,:,:) =  diag(sigma);
           end
           
           for i=1:C
                hij_sum =  sum(FG_hij(i,:));
                FG_mu(i,:) = FG_hij(i,:)* t_D_FG ./ hij_sum;
                sigma = FG_hij(i,:) * (t_D_FG-FG_mu(i,:)).^2  ./ hij_sum;
                 % In case the convariance diagnal elements are zeros
                sigma(find(sigma<1e-4))=1e-4;
                conv_FG(i,:,:) =  diag(sigma);
           end
            pi_BG = sum(BG_hij,2)/count_BG;
            pi_FG = sum(FG_hij,2)/count_FG;

            arg_BG(mix).pi = pi_BG;
            arg_BG(mix).mu =  BG_mu;
            arg_BG(mix).cov =  conv_BG;
            
            arg_FG(mix).pi = pi_FG;
            arg_FG(mix).mu =  FG_mu;
            arg_FG(mix).cov =  conv_FG;
        end
    end
    
for dim=dims
    dirname = ['prmblemA/',int2str(dim)];
    if ~exist(dirname, 'dir')
       mkdir(dirname)
    end
    % Start to generate mask for each pair of classifier
    for BG_id =1:mix_num
        for FG_id=1:mix_num
            disp(['Dim=',int2str(dim),'  Generating images',' FG=',int2str(FG_id),' BG=',int2str(BG_id)])
            FG=arg_FG(FG_id);
            BG=arg_BG(BG_id);
            brol = rol-7;
            bcol = col-7;
            dsize = size(features,1);
            mask = zeros(brol,bcol);
            for b=0:dsize-1
                j = mod(b,bcol);
                i =(b-j)/(bcol);
                score_BG=zeros(C,1);
                score_FG=zeros(C,1);
                % Computer likelihood for each component
                for classfier = 1:C
                    FG_cor_t = reshape(FG.cov(classfier,1:dim,1:dim),[dim dim]);
                    BG_cor_t = reshape(BG.cov(classfier,1:dim,1:dim),[dim dim]);
                    FG_mu_t = FG.mu(classfier,1:dim);
                    BG_mu_t = BG.mu(classfier,1:dim);
                    pi_FG_t = FG.pi(classfier);
                    pi_BG_t = BG.pi(classfier);
                    score_FG(classfier)=MVNB(FG_cor_t,FG_mu_t,features(b+1,1:dim),P_Y(2)*pi_FG_t);
                    score_BG(classfier)=MVNB(BG_cor_t,BG_mu_t,features(b+1,1:dim),P_Y(1)*pi_BG_t);
                end
                
                if min(score_FG)<min(score_BG)
                        mask(i+1,j+1)=255;
                end
            end
            mask_name=['prmblemA/',int2str(dim),'/cheetah_premask_EM',int2str(dim),'mix',int2str(FG_id),int2str(BG_id),'.png'];
            imwrite(mask,mask_name);
        end
    end
end

function prob = guassian(x,mu,C,d)
    prob = exp(1/2*((x-mu)*inv(C)*(x-mu)'))/((2*pi)^(d/2)*sqrt(det(C)));
end
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
    img=padarray(img,[3,3],'symmetric','both');
    img=padarray(img,[1,1],'symmetric','post');
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