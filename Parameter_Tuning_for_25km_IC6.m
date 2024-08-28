clc;clear all;close all;

rng(100)

%% get coordinate list:
geo_filename = '/Volumes/Pruina_External_Elements/WPO_SnowFration_Research/Data_repository/25km/NoahMP_Setup_File/HRLDAS_setup_25km_wus_SnowPixels_1D.nc';
latvec = ncread(geo_filename,'XLAT');
lonvec = ncread(geo_filename,'XLONG');
lonvec = wrapTo180(lonvec);
idx_east = find(lonvec>-104.4);
%% define data input directories for reference snow data
SNODAS_dir = '/Volumes/Pruina_External_Elements/WPO_SnowFration_Research/Data_repository/Reference_Snow_Data/Reference_Snow_Data_25km/SNODAS/'; %FILES BY WY
MODSCAG_dir = '/Volumes/Pruina_External_Elements/WPO_SnowFration_Research/Data_repository/Reference_Snow_Data/Reference_Snow_Data_25km/MODSCAG/'; %FILE BY CY

%% get graining data from 2004-2012 WYs:
SCF_train=[];
SWE_train=[];
SNOWH_train=[];
for WY=2004:2012
    SNODAS_filename = sprintf('SNODAS_WY%d.nc',WY);
    MODSCAG_filename = sprintf('MODSCAG_SCF_WY%d.nc',WY);
    
    SNODAS_SWE = ncread([SNODAS_dir,SNODAS_filename],'SWE');
    SNODAS_SNOWH = ncread([SNODAS_dir,SNODAS_filename],'SNOWH');

    MODSCAG_SCF = ncread([MODSCAG_dir,MODSCAG_filename],'SCF');

    SCF_train=[SCF_train,MODSCAG_SCF];
    SWE_train=[SWE_train,SNODAS_SWE];
    SNOWH_train=[SNOWH_train,SNODAS_SNOWH];
end
%trim to WUS bound:
SCF_train(idx_east,:)=[];
SWE_train(idx_east,:)=[];
SNOWH_train(idx_east,:)=[];

%% convert to correct units for SCF NY07 formula:
SCF_train = SCF_train./100; %frac
SNOWH_train = SNOWH_train./1000;%m
DEN_train = SWE_train./SNOWH_train;

%remove unrealistic values and instances of 0 snow from MODSCAG or SNODAS:
idx_nan=find(SNOWH_train>20 | DEN_train>900 | DEN_train<50 | SCF_train==0 | SNOWH_train == 0);
SCF_train(idx_nan)=NaN;
SNOWH_train(idx_nan)=NaN;
DEN_train(idx_nan)=NaN;

idx=find(isnan(SCF_train) | isnan(SNOWH_train) | isnan(DEN_train));
SCF_train(idx)=NaN;
SNOWH_train(idx)=NaN;
DEN_train(idx)=NaN;

%% define indices to allow organization of data by water year:
WY_dates = datenum([2009 10 1]):datenum([2010 9 30]);
WY_datevecs = datevec(WY_dates);
WY_datevecs = WY_datevecs(:,2:3);

datelist = datenum([2004 1 1]):datenum([2012 12 31]);
datevecs = datevec(datelist);
unique_years = unique(datevecs(:,1));
start_year = min(unique_years);
end_year = max(unique_years);

%define ablation and accumulation periods across total tuning period:
idx_accum_all = find(datevecs(:,2)==11 | datevecs(:,2)==12 | datevecs(:,2)==1 | datevecs(:,2)==2|datevecs(:,2)==3);
idx_ablat_all = find(datevecs(:,2)>=4 & datevecs(:,2)<=7);

%define indices for accumulaiton and ablation periods based on WY calendar:
idx_accum = 32:182; %Nov1 - March 31
idx_ablat = 183:304; %April 1 -July 31

[u,~,j] = unique(datevecs(:,2:3),'rows');
[ia,ib] = ismember(WY_datevecs,u,'rows');

%define dimesnion of training data:
S = size(SCF_train);

%define bias tolerances:
ablat_tol = 0.004;
accum_tol = 0.004;

%initialize variables:
iter = 0;
max_iter = 300;
Param_increment = 6*10^(-4);
ablat_bias_opt = 999; %initialize mean abalation period bias (based on aggregation in the WY calendar)
accum_bias_opt = 999; %initialize mean abalation period bias (based on aggregation in the WY calendar)

%~for diagnostics~
Store_stats_opt=[];
store_opt_params=[];

%define initial guess for NY07 parameters:
opt_param_original = [0.008 1];

%Define the SCF equation and training data:
func_original= @(Params,X) tanh( X(:,1) ./ ( Params(1) .* ( (X(:,2)./100).^(Params(2)) ) ) );
X1_original = [SNOWH_train(:),DEN_train(:)];

%perform calcs and regrouping that only needs to be done 1 time (ie on the training data)
Y1 = SCF_train(:);
mean_spatial_obs_store = nanmean(SCF_train,1);
mean_spatial_obs_store = accumarray(j,mean_spatial_obs_store,[],@nanmean);
mean_pixel_SCF_obs = nanmean(SCF_train,2);
SCF_train_accum = SCF_train(:,idx_accum_all);
SCF_train_ablat = SCF_train(:,idx_ablat_all);

%% Perform the parameter tuning procedure:
while abs(ablat_bias_opt) > ablat_tol || abs(accum_bias_opt) > accum_tol
    %count this iteration
    iter = iter+1 

    %compute SCF from NY07 formula using the current parameters 
    curve_fit_original_opt  =  func_original(opt_param_original,X1_original);
    
    %reshape the model SCF for later use: 
    curve_fit_original_opt_gridded = reshape(curve_fit_original_opt,[S(1) , S(2)]);

    %calculate bias and RMSE across all data:
    bias_original = curve_fit_original_opt - Y1;
    RMSE_original = calc_RMSE(curve_fit_original_opt,Y1);

    %calculate biases organized by pixel:
    original_bias_grid = curve_fit_original_opt_gridded - SCF_train;

    %compute the mean pixel SCF & bias from model:
    mean_pixel_SCF_original = nanmean(curve_fit_original_opt_gridded,2);
    mean_pixel_original_bias = mean_pixel_SCF_original - mean_pixel_SCF_obs;

    %group model data by accumulation and ablation period across the total time series
    curve_fit_original_opt_gridded_accum = curve_fit_original_opt_gridded(:,idx_accum_all);
    curve_fit_original_opt_gridded_ablat = curve_fit_original_opt_gridded(:,idx_ablat_all);
    
    %calculate RMSE & bias for accumulation and ablation periods for total domain
    RMSE_accum = calc_RMSE(curve_fit_original_opt_gridded_accum(:),SCF_train_accum(:));
    RMSE_ablat = calc_RMSE(curve_fit_original_opt_gridded_ablat(:),SCF_train_ablat(:));

    bias_total_accum = nanmean(curve_fit_original_opt_gridded_accum(:) - SCF_train_accum(:)); 
    bias_total_ablat = nanmean(curve_fit_original_opt_gridded_ablat(:) - SCF_train_ablat(:)); 

    %%  calculate the spatially averaged multiyear timeseries for the model data:
    mean_spatial_opt = nanmean(curve_fit_original_opt_gridded,1);
    mean_spatial_opt = accumarray(j,mean_spatial_opt,[],@nanmean);
    mean_spatial_obs = mean_spatial_obs_store;
    
    mean_spatial_opt = mean_spatial_opt(ib);
    mean_spatial_obs = mean_spatial_obs(ib);
    
    %seperate accumulation and ablation period data:
    accum_bias_opt = nanmean(mean_spatial_opt(idx_accum) - mean_spatial_obs(idx_accum));
    ablat_bias_opt = nanmean(mean_spatial_opt(idx_ablat) - mean_spatial_obs(idx_ablat));
    
    %apply parameter increments that adjust the SCF parameters based on the seasonal biases

    if accum_bias_opt>0 && ablat_bias_opt>0 %too large SCF during accum & ablation:
        opt_param_original(1) = opt_param_original(1)+Param_increment; %increase Fsno
        opt_param_original(2) = opt_param_original(2)+Param_increment; %increase m
    elseif accum_bias_opt>0 && ablat_bias_opt<0 %too large SCF during accum but too small during ablation:
        if abs(accum_bias_opt) > abs(ablat_bias_opt) %if accumulation period bias is larger than ablaiton period bias
            %impose proportionally larger adjustment to Fsno:
            opt_param_original(1) = opt_param_original(1)+Param_increment./ max((abs(ablat_bias_opt)/abs(accum_bias_opt)),0.1); %increase Fsno
            if abs(ablat_bias_opt) > ablat_tol %if ablation period bias is larger than tolerance, also adjust m:
                opt_param_original(2) = opt_param_original(2)-Param_increment; %decrease m
            end
        else %if the ablation period bias is larger than accumulation bias:
            if abs(accum_bias_opt) > accum_tol %if accumulation period bias is larger than tolerance, also adjust Fsno:
                opt_param_original(1) = opt_param_original(1)+Param_increment;
            end
            %impose proportionally larger adjustment to m:
            opt_param_original(2) = opt_param_original(2)-Param_increment./max((abs(accum_bias_opt)/abs(ablat_bias_opt)),0.1);
        end
    elseif accum_bias_opt<0 && ablat_bias_opt>0 %too small SCF during accum but too large during ablation:
        if abs(accum_bias_opt) > abs(ablat_bias_opt) %if the accumulation bias is larger than the ablation bias
            %impose proportionally larger adjustment to Fsno:
            opt_param_original(1) = opt_param_original(1)-Param_increment./ max((abs(ablat_bias_opt)/abs(accum_bias_opt)),0.1);
            if abs(ablat_bias_opt) > ablat_tol %if ablation period bias is larger than tolerance, also adjust m:
                opt_param_original(2) = opt_param_original(2)+ Param_increment ;
            end
        else %if the ablation period bias is larger than accumulation bias:
            if abs(accum_bias_opt) > accum_tol %if accumulation period bias is larger than tolerance, also adjust Fsno:
                opt_param_original(1) = opt_param_original(1)-Param_increment ;
            end
             %impose proportionally larger adjustment to m:
            opt_param_original(2) = opt_param_original(2)+ Param_increment./ max((abs(accum_bias_opt)/abs(ablat_bias_opt)),0.1);
        end
    elseif accum_bias_opt<0 && ablat_bias_opt<0  %too small SCF in both accum and ablat periods:
        opt_param_original(1) = opt_param_original(1)-Param_increment; %decrease Fsno
        opt_param_original(2) = opt_param_original(2)-Param_increment; %decrease m
    end

    %store the ending parameters derived from each iteration:
    disp('ending opt params')
    store_opt_params = [store_opt_params;opt_param_original];
    
    %store the metrics from each iteration:
    disp('ending opt biases')
    Store_stats_opt = [Store_stats_opt; accum_bias_opt , ablat_bias_opt , nanmean(bias_original), RMSE_original,RMSE_accum,RMSE_ablat,bias_total_accum,bias_total_ablat];

    %max iter criteria for non-convergence:
    if iter>max_iter
        disp('too many iters')
        break
    end

    %if RMSE is increasing while biases are converged (oscillating), exit loop
    if iter > 3
        delta_RMSE = Store_stats_opt(end,4) - Store_stats_opt(end-1,4); %this should be positive
        delta_bias1_accum = Store_stats_opt(end,1) - Store_stats_opt(end-1,1); %this should be negative
        delta_bias2_accum = Store_stats_opt(end-1,1) - Store_stats_opt(end-2,1); %this should be positive
        delta_bias1_ablat = Store_stats_opt(end,2) - Store_stats_opt(end-1,2); %this should be negative
        delta_bias2_ablat = Store_stats_opt(end-1,2) - Store_stats_opt(end-2,2); %this should be positive
        if delta_RMSE> 0 && delta_bias1_accum<0 && delta_bias2_accum>0 && delta_bias1_ablat<0 && delta_bias2_ablat>0
            disp('RMSE is increasing while biases are converged')
            break
        end
    end
end

%after above loop, iteratively update Fsno parameter to remove any systematic bias across the total dataset:

%bias across all data
curve_fit_original_opt  =  func_original(opt_param_original,X1_original);
bias_original = curve_fit_original_opt - Y1;
opt_bias = nanmean(bias_original); 

%SCF formula:
func_original= @(Params,X) tanh( X(:,1) ./ ( Params(1) .* ( (X(:,2)./100).^(Params(2)) ) ) );
%bias tolerance:
bias_tol=0.01;
%the final parameters from the end of the prior loop:
opt_params = store_opt_params(end,:);
%initialize data:
iter = 0;
converge_criteria=0;
converge_counter=0;
increment_size = 0.001;
Store_bias_opt_new = [];
%loop through to remove biases:
while abs(opt_bias) > bias_tol || converge_criteria == 0
    iter = iter+1
    if opt_bias < 0 %if tuned model bias is negative
        opt_params(1) = opt_params(1) -increment_size; %decrease Fsno
    else
        opt_params(1) = opt_params(1) +increment_size; %increase Fsno
    end
    curve_fit_opt  =  func_original(opt_params,X1_original); %recalculate the box model SCF
    opt_bias = curve_fit_opt - Y1; %and corresponding bias

    curve_fit_original_opt_gridded = reshape(curve_fit_opt,[S(1) , S(2)]); %reshape to calculate biases across pixels
    %seperate data by accumulation and ablation periods:
    curve_fit_original_opt_gridded_accum = curve_fit_original_opt_gridded(:,idx_accum_all);
    curve_fit_original_opt_gridded_ablat = curve_fit_original_opt_gridded(:,idx_ablat_all);

    SCF_train_accum = SCF_train(:,idx_accum_all);
    SCF_train_ablat = SCF_train(:,idx_ablat_all);
    
    %calculate RMSE and biases:
    RMSE_accum = calc_RMSE(curve_fit_original_opt_gridded_accum(:),SCF_train_accum(:));
    RMSE_ablat = calc_RMSE(curve_fit_original_opt_gridded_ablat(:),SCF_train_ablat(:));
    RMSE_original = calc_RMSE(curve_fit_opt,Y1);

    bias_total_accum = nanmean(curve_fit_original_opt_gridded_accum(:) - SCF_train_accum(:)); 
    bias_total_ablat = nanmean(curve_fit_original_opt_gridded_ablat(:) - SCF_train_ablat(:)); 

    %calculate the spatial multiyear mean bias:
    mean_spatial_opt = nanmean(curve_fit_original_opt_gridded,1);
    mean_spatial_opt = accumarray(j,mean_spatial_opt,[],@nanmean);
    mean_spatial_obs = mean_spatial_obs_store;

    mean_spatial_opt = mean_spatial_opt(ib);
    mean_spatial_obs = mean_spatial_obs(ib);
  
    accum_bias_opt = nanmean(mean_spatial_opt(idx_accum) - mean_spatial_obs(idx_accum));
    ablat_bias_opt = nanmean(mean_spatial_opt(idx_ablat) - mean_spatial_obs(idx_ablat));
    opt_bias = nanmean(opt_bias);

    %storing stats for background diagnostics:
    Store_stats_opt = [Store_stats_opt; accum_bias_opt , ablat_bias_opt , opt_bias, RMSE_original,RMSE_accum,RMSE_ablat,bias_total_accum,bias_total_ablat];
    Store_bias_opt_new = [Store_bias_opt_new;opt_bias];
    store_opt_params = [store_opt_params;opt_params];

    %if the bias across all data begins oscilating, start counting towards convergence:
    if min(Store_bias_opt_new) < 0 && max(Store_bias_opt_new) >0
        converge_counter = converge_counter+1;
        increment_size = increment_size*0.95; %reduce increment size in each iteration from hereon
    end

    if converge_counter >= 30 && abs(opt_bias) < bias_tol 
        converge_criteria = 1; %to note that the convergence criteria has been met allowing 30 oscillations
    end

    sprintf('ending bias: %.4f',round(opt_bias,4))

end

dlmwrite('/Users/abolafia/WPO_SnowFration_Research/Data/opt_params_original_test_25km.csv',store_opt_params,'delimiter',',','precision',8);
dlmwrite('/Users/abolafia/WPO_SnowFration_Research/Data/skill_scores_by_iter_original_test_25km.csv',Store_stats_opt,'delimiter',',','precision',8);

% plot results from parameter tuning for diagnostics:
skill_scores = csvread('/Users/abolafia/WPO_SnowFration_Research/Data/skill_scores_by_iter_original_test_25km.csv');
opt_params = csvread('/Users/abolafia/WPO_SnowFration_Research/Data/opt_params_original_test_25km.csv');
skill_scores_abs = abs(skill_scores);

f=figure;
f.Position = [-1919        -226        1731        1023];

subplot(3,3,1)
plot(skill_scores(:,7),'-.','color','k','markersize',6)
ylabel('accum bias total','fontsize',24)
set(gca,'fontsize',22)
grid on

subplot(3,3,2)
plot(skill_scores(:,8),'-.','color','k','markersize',6)
ylabel('ablat bias total','fontsize',24)
set(gca,'fontsize',22)
grid on

subplot(3,3,3)
plot(skill_scores(:,3),'-.','color','k','markersize',6)
ylabel('total bias','fontsize',24)
set(gca,'fontsize',22)
grid on

subplot(3,3,4)
plot(skill_scores(:,4),'-.','color','k','markersize',6)
ylabel('total rmse','fontsize',24)
set(gca,'fontsize',22)
grid on

subplot(3,3,5)
plot(skill_scores(:,5),'-.','color','k','markersize',6)
ylabel('accum rmse','fontsize',24)
set(gca,'fontsize',22)
grid on

subplot(3,3,6)
plot(skill_scores(:,6),'-.','color','k','markersize',6)
ylabel('ablat rmse','fontsize',24)
set(gca,'fontsize',22)
grid on

subplot(3,3,7)
hold on
plot(skill_scores(:,1),'-.','color','k','markersize',6)
ylabel({'spatial multiyear avg';'accumulation bias'},'fontsize',24)
set(gca,'fontsize',22)
grid on

subplot(3,3,8)
plot(skill_scores(:,2),'-.','color','k','markersize',6)
ylabel({'spatial multiyear avg';'ablation bias'},'fontsize',24)
set(gca,'fontsize',22)
grid on


