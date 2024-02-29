winlength = 4;
parameters.workingdir = '../RESULTS/ccf_auto/';

parameters.ccfpath = [parameters.workingdir,'ccf/'];
parameters.figpath = [parameters.workingdir,'figs/'];
parameters.seis_path = [parameters.workingdir,'seismograms/'];
parameters.logpath = [parameters.workingdir,'ccf_log/'];

% ------ Build File Structure: cross-correlations -------
ccf_path = parameters.ccfpath;
ccf_winlength_path = [ccf_path,'window',num2str(winlength),'hr/'];
ccf_singlestack_path = [ccf_winlength_path,'single/'];
ccf_daystack_path = [ccf_winlength_path,'dayStack/'];
ccf_monthstack_path = [ccf_winlength_path,'monthStack/'];
ccf_fullstack_path = [ccf_winlength_path,'fullStack/'];

if ~exist(ccf_path)
    mkdir(ccf_path)
end
if ~exist(ccf_winlength_path)
    mkdir(ccf_winlength_path)
end
if ~exist(ccf_singlestack_path)
    mkdir(ccf_singlestack_path)
end
if ~exist(ccf_daystack_path)
    mkdir(ccf_daystack_path)
end
if ~exist(ccf_monthstack_path)
    mkdir(ccf_monthstack_path)
end
if ~exist(ccf_fullstack_path)
    mkdir(ccf_fullstack_path)
end
if ~exist(parameters.logpath)
    mkdir(parameters.logpath)
end

txtpathR = [parameters.workingdir,'text_output/RayleighResponse_R/'];
txtpathT = [parameters.workingdir,'text_output/LoveResponse/'];
txtpathZ = [parameters.workingdir,'text_output/RayleighResponse/'];
if ~exist(txtpathZ)
    mkdir(txtpathZ)
end
if ~exist(txtpathR)
    mkdir(txtpathR)
end
if ~exist(txtpathT)
    mkdir(txtpathT)
end

log_path = [parameters.workingdir,'ccf_log/'];
if ~exist(log_path)
    mkdir(log_path)
end

PATHS = {ccf_singlestack_path; ccf_daystack_path; ccf_monthstack_path; ccf_fullstack_path};

for ipath = 1:length(PATHS)
    ccfR_path = [PATHS{ipath},'ccfRR/'];
    ccfT_path = [PATHS{ipath},'ccfTT/'];
    ccfZ_path = [PATHS{ipath},'ccfZZ/'];
    if ~exist(ccfR_path)
        mkdir(ccfR_path);
    end
    if ~exist(ccfT_path)
        mkdir(ccfT_path);
    end
    if ~exist(ccfZ_path)
        mkdir(ccfZ_path);
    end
end

% Build File Structure: figures
figpath = parameters.figpath;
fig_winlength_path = [figpath,'window',num2str(winlength),'hr/'];
if ~exist(figpath)
    mkdir(figpath);
end
if ~exist(fig_winlength_path)
    mkdir(fig_winlength_path);
end

% Build File Structure: windowed seismograms
seis_path = parameters.seis_path;
seis_winlength_path = [seis_path,'window',num2str(winlength),'hr/'];
if ~exist(seis_path)
    mkdir(seis_path);
end
if ~exist(seis_winlength_path)
    mkdir(seis_winlength_path);
end

parameters.figpath = [parameters.workingdir,'figs/'];