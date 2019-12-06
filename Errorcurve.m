%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Compute and Plot the POE curve %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dataset index
BG_id =4;

D_BG= TrainsampleDCT_BG;
D_FG= TrainsampleDCT_FG;

count_BG = size(D_BG,1);
count_FG = size(D_FG,1);

% ML estimates for the class priors
P_Y=[count_BG  count_FG]/(count_FG+count_BG);
dims = [1,2,4,8,16,24,32,40,48];
size_dim = size(dims,2);
mix = 5;
errors= zeros(mix ,size_dim);

% Compute POE for predictive solution
for FG_id = 1:mix
	for i=1:size_dim
        dim=dims(i);
		mask_name=['prmblemA/',int2str(dim),'/cheetah_premask_EM',int2str(dim),'mix',int2str(FG_id),int2str(BG_id),'.png'];
		mask = imread(mask_name);
		error = compute_error(mask,P_Y);
		errors(FG_id,i)=error;
	end
end
% Curve Plot
y_min = min(min(errors));
y_max = max(max(errors));
deta = (y_max-y_min)/8;
figure;

plot(dims, errors(1,:),dims, errors(2,:),dims, errors(3,:),dims,errors(4,:),dims,errors(5,:));
title_name=['Probability of Error For Background ',int2str(BG_id)];
title(title_name)
xlabel('Dimension')
ylabel('Probability of Error')
legend({'FG1','FG2','FG3','FG4','FG5'},'Location','southeast')
ylim([y_min-deta y_max+deta])
saveas(gcf,['plot/POE_BG',int2str(BG_id),'.png']);

function error = compute_error(mask,PY)
    % compute probability of error
    mask_gt = imread('cheetah_mask.bmp');
    mask_gt = double(mask_gt);
    mask = double(mask);
    [m,n] = size(mask);
    TN = 0;
    FN=0;
    TP =0;
    FP =0;
    for i=1:m
        for j=1:n
            
            % calculate the TN
             if mask_gt(i,j)==255&&mask(i,j)==0
                TN=TN+1;
             end
             % calculate the FN
             if (mask_gt(i,j)==0)&&(mask(i,j)==0)
                 FN=FN+1;
             end
             % calculate the TP
             if (mask_gt(i,j)==255)&&(mask(i,j)==255)
                TP =TP+1;
             end
             % calculate the FP
             if mask_gt(i,j)==0&&mask(i,j)==255
                FP =FP+1;
             end
        end
    end
   TP_rate  = TP/(TP+TN);
    FP_rate= FP/(FP+FN);
    error = (1-TP_rate)*PY(2)+ FP_rate*PY(1);
end