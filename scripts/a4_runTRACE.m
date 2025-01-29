% ---- Adapted from 
% M.A. Muñoz and K. Smith-Miles. Instance Space Analysis: A toolkit for the assessment of algorithmic power. andremun/InstanceSpace on Github. Zenodo, DOI:10.5281/zenodo.4484107, 2020.
% ----

function a4b_runTRACE(paramPath, paramIdx, rfol, proj_file)

% rfol for mda folder, '' otherwise
% proj_file for proj coords file


params = readtable(paramPath,"TextType","string");
% paramIdx = paramIdx + 1;

%% Opts to change --------------------
outfolder = params.outfolder{paramIdx};     

opts.perf.MaxPerf = params.maxPerf(paramIdx);   % false    % false;          % True if Y is a performance measure to maximize, False if it is a cost measure to minimise.
opts.perf.AbsPerf = params.absPerf(paramIdx);   % false;           % True if an absolute performance measure, False if a relative performance measure
opts.perf.epsilon = params.epsilon(paramIdx);   % 0.0;           % Threshold of good performance

opts.perf.betaThreshold = 0.55;     % Beta-easy threshold
opts.auto.preproc = true;           % Automatic preprocessing on. Set to false if you don't want any preprocessing
opts.bound.flag = false;             % Bound the outliers. True if you want to bound the outliers, false if you don't
opts.norm.flag = true;              % Normalize/Standarize the data. True if you want to apply Box-Cox and Z transformations to stabilize the variance and scale N(0,1)

opts.trace.usesim = false;           % Use the actual or simulated data to calculate the footprints
opts.trace.PI = 0.55;               % Purity threshold
opts.trace.Trace2 = false;      %    % Use Trace2 instead of TRACE
opts.trace.prior = [0.6,0.4];       % Trace2 Prior Weighting


try
    runTrace([outfolder rfol],opts, proj_file);
catch ME
    disp('EOF:ERROR');
    rethrow(ME)
end
end

%% Function to run TRACE ========================================
function runTrace(outfolder,opts, proj_file)    

    scriptfcn;

    datapathTr = [outfolder 'processed_train.csv'];
    datapathTest = [outfolder 'processed_test.csv'];
    proj_path = [outfolder proj_file];
    
    footpath = [outfolder 'footprint/'];
    mkdir(footpath);

    % -------------------------------------------------------------------------
    % Reading the projection data
    projections = readtable(proj_path,'TextType','string');
    projNames = unique(projections.proj);
    
    Xbar_train = readtable(datapathTr,'TextType','string');
    Xbar_test = readtable(datapathTest,'TextType','string');
    varlabels = Xbar_train.Properties.VariableNames;
    isalgo = strncmpi(varlabels,'algo_',5);
    isbin = strncmpi(varlabels,'bin_',4);  %%%

    algoNames = varlabels(isalgo);

    B_train = Xbar_train{:,"best"};
    B_test = Xbar_test{:,"best"};
    Ybin_train = logical(Xbar_train{:,isbin});
    Ybin_test = logical(Xbar_test{:,isbin});
    
    nalgos = size(algoNames,2);    
    nclass = length(unique(vertcat(B_train,B_test)));

    disp([nalgos nclass]);

    if ismember('ANY', Xbar_train{:,"best"})
        algoNames{end+1} = 'ANY';
    end

    if nclass<3  
        % if less than 2 classes, cannot calculate footprints for LDA
        projNames(strcmp(projNames,'LDA')) = [];
    end

    
    % -------------------------------------------------------------------------
    numGoodAlgos_Tr = sum(Ybin_train,2);
    beta_train = numGoodAlgos_Tr>(opts.perf.betaThreshold*nalgos);

    numGoodAlgos_Test = sum(Ybin_test,2);
    beta_test = numGoodAlgos_Test>(opts.perf.betaThreshold*nalgos);

    % -------------------------------------------------------------------------
    % Calculating the algorithm footprints.

    allBestPre = cell(1,size(projNames,2));
    allBest = cell(1,size(projNames,2));
    allGood = cell(1,size(projNames,2));
    performanceSummary_train = cell(1,size(projNames,2));
    performanceSummary_test = cell(1,size(projNames,2));

    for j=1:length(projNames)        

        Z_train = projections{projections.group == "train" & projections.proj == projNames{j}, ...
            {'Z1','Z2'}};
        trace = TRACE(Z_train, Ybin_train, B_train, beta_train, algoNames, opts.trace); % changed from P to B
        
        if projNames{j} == "PLSRda" || projNames{j} == "MDA"
            scriptpng(Z_train,Ybin_train, B_train, trace, algoNames,projNames{j}, footpath);
        end
        

        Z_test = projections{projections.group == "test" & projections.proj == projNames{j}, ...
            {'Z1','Z2'}};
        trace_test = TRACEtest(trace,Z_test,Ybin_test,B_test,beta_test,algoNames);

        bestFootprints = cell(1,nclass);
        preBestFootprints = cell(1,nclass);
        goodFootprints = cell(1,nalgos);

        
        for i=1:nclass
            if isfield(trace.best{i},'polygon') && ~isempty(trace.best{i}.polygon)
                writeArray2CSV(trace.best{i}.polygon.Vertices, ...
                    {'Z1','Z2'},...
                    makeBndLabels(trace.best{i}.polygon.Vertices),...
                    [footpath 'bestFP_' projNames{j} '-' algoNames{i} '.csv']);
            end

            if isfield(trace.bestPre{i},'polygon') && ~isempty(trace.bestPre{i}.polygon)
                F = array2table(trace.bestPre{i}.polygon.Vertices,'VariableNames', ...
                    {'Z1','Z2'});
                F.algo = repmat(string(algoNames{i}),size(F,1),1);
                F.proj = repmat(string(projNames{j}),size(F,1),1);

                preBestFootprints{i} = F;
                
            end

            if isfield(trace.best{i},'polygon') && ~isempty(trace.best{i}.polygon)
                F = array2table(trace.best{i}.polygon.Vertices,'VariableNames', ...
                    {'Z1','Z2'});
                F.algo = repmat(string(algoNames{i}),size(F,1),1);
                F.proj = repmat(string(projNames{j}),size(F,1),1);

                bestFootprints{i} = F;
                
            end
        end
        for i=1:nalgos
            if isfield(trace.good{i},'polygon') && ~isempty(trace.good{i}.polygon)
                writeArray2CSV(trace.good{i}.polygon.Vertices, ...
                    {'Z1','Z2'},...
                    makeBndLabels(trace.good{i}.polygon.Vertices),...
                    [footpath 'goodFP_' projNames{j} '-' algoNames{i} '.csv']);
            end

            if isfield(trace.good{i},'polygon') && ~isempty(trace.good{i}.polygon)
                Fg = array2table(trace.good{i}.polygon.Vertices,'VariableNames', ...
                    {'Z1','Z2'});
                Fg.algo = repmat(string(algoNames{i}),size(Fg,1),1); 
                Fg.proj = repmat(string(projNames{j}),size(Fg,1),1);
                goodFootprints{i} = Fg;
        
            end
        end
        bestFootprints = vertcat(bestFootprints{:});
        allBest{j} = bestFootprints;

        preBestFootprints = vertcat(preBestFootprints{:});
        allBestPre{j} = preBestFootprints;
        
        goodFootprints = vertcat(goodFootprints{:});
        allGood{j} = goodFootprints;

        perfLabs = trace.summary(1,:);
        perfLabs{1} = 'algo';    
        perfSumm = cell2table(trace.summary(2:end,:), 'VariableNames',perfLabs);    
        perfSumm.proj = repmat(string(projNames{j}),size(perfSumm,1),1);
        performanceSummary_train{j} = perfSumm;

        perfLabsTest = trace_test.summary(1,:);
        perfLabsTest{1} = 'algo';
        perfSummTest = cell2table(trace_test.summary(2:end,:), 'VariableNames',perfLabsTest);
        perfSummTest.proj = repmat(string(projNames{j}),size(perfSummTest,1),1);
        performanceSummary_test{j} = perfSummTest;
        
        
    end

    allBest = vertcat(allBest{:});
    writetable(allBest,[footpath '/best_pts.csv']);

    allBestPre = vertcat(allBestPre{:});
    writetable(allBestPre,[footpath '/bestPre_pts.csv']);

    allGood = vertcat(allGood{:}); 
    writetable(allGood,[footpath '/good_pts.csv']);

    performanceSummary_train = vertcat(performanceSummary_train{:});
    writetable(performanceSummary_train,[footpath '/performanceTrain.csv']);

    performanceSummary_test = vertcat(performanceSummary_test{:});
    writetable(performanceSummary_test,[footpath '/performanceTest.csv']);

    % -------------------------------------------------------------------------
end
