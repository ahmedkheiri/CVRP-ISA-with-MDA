% ---- Adapted from 
% M.A. Muñoz and K. Smith-Miles. Instance Space Analysis: A toolkit for the assessment of algorithmic power. andremun/InstanceSpace on Github. Zenodo, DOI:10.5281/zenodo.4484107, 2020.
% ----

function a2_runSIFTED(paramPath, paramIdx, sk, md)

params = readtable(paramPath,"TextType","string");
% paramIdx = paramIdx + 1;

%% Opts to change --------------------
outfolder = [params.outfolder{paramIdx} md];     

opts.perf.MaxPerf = params.maxPerf(paramIdx);   % false    % false;          % True if Y is a performance measure to maximize, False if it is a cost measure to minimise.
opts.perf.AbsPerf = params.absPerf(paramIdx);   % false;           % True if an absolute performance measure, False if a relative performance measure
opts.perf.epsilon = params.epsilon(paramIdx);   % 0.0;           % Threshold of good performance

opts.sifted.K = sk;                 % Number of final features. Ideally less than 10.

opts.perf.betaThreshold = 0.55;     % Beta-easy threshold
opts.auto.preproc = true;           % Automatic preprocessing on. Set to false if you don't want any preprocessing
opts.bound.flag = false;             % Bound the outliers. True if you want to bound the outliers, false if you don't
opts.norm.flag = true;              % Normalize/Standarize the data. True if you want to apply Box-Cox and Z transformations to stabilize the variance and scale N(0,1)

opts.sifted.flag = true;            % Automatic feature selectio on. Set to false if you don't want any feature selection.
opts.sifted.rho = 0.3;              % Minimum correlation value acceptable between performance and a feature. Between 0 and 1
opts.sifted.NTREES = 50;            % Number of trees for the Random Forest (to determine highest separability in the 2-d projection)
opts.sifted.MaxIter = 1000;
opts.sifted.Replicates = 100;



try
    %% Build model
    datapath = strcat(outfolder, 'processed_train.csv');    
    model = runSifted(datapath,opts,outfolder);

    scriptfcn;
    % model = load(strcat(outfolder, 'model.mat'));
    sel_feats = strcat('feature_', model.data.featlabels);
    writeCell2CSV(sel_feats, compose('f%d', 1:length(sel_feats)), {'feature'}, strcat(outfolder, 'selected_features.csv'));
    
    
catch ME
    disp('EOF:ERROR');
    rethrow(ME)
end
end

%% Helper functions ----------------------------------------------
function model = runSifted(datapath,opts,rootdir)
    
    scriptfcn;
    startProcess = tic;
    % -------------------------------------------------------------------------
    rng(111);
    
    %% Load the data
    disp('-------------------------------------------------------------------------');
    disp('-> Loading the data.');
    Xbar = readtable(datapath);
    varlabels = Xbar.Properties.VariableNames;
    isname = strcmpi(varlabels,'instances');
    isfeat = strncmpi(varlabels,'feature_',8);
    isalgo = strncmpi(varlabels,'algo_',5);
    issource = strcmpi(varlabels,'source');
    isbest = strcmpi(varlabels,'best');  %%%
    isbin = strncmpi(varlabels,'bin_',4);  %%%

    model.data.instlabels = Xbar{:,isname};
    if isnumeric(model.data.instlabels)
        model.data.instlabels = num2cell(model.data.instlabels);
        model.data.instlabels = cellfun(@(x) num2str(x),model.data.instlabels,'UniformOutput',false);
    end
    if any(issource)
        model.data.S = categorical(Xbar{:,issource});
    end
    model.data.X = Xbar{:,isfeat};
    model.data.Y = Xbar{:,isalgo};
    model.data.Ybest = Xbar{:,isbest};
    model.data.Ybin = Xbar{:,isbin};
    
    model.data.featlabels = varlabels(isfeat);
    model.data.algolabels = varlabels(isalgo);
    
    % -------------------------------------------------------------------------
    % Storing the raw data for further processing, e.g., graphs
    model.data.Xraw = model.data.X;
    model.data.Yraw = model.data.Y;
    % -------------------------------------------------------------------------
    % Removing the template data such that it can be used in the labels of
    % graphs and figures.
    model.data.featlabels = strrep(model.data.featlabels,'feature_','');
    model.data.algolabels = strrep(model.data.algolabels,'algo_','');
    
    % -------------------------------------------------------------------------
    %% Run SIFTED for feature selection
    % Automated feature selection.
    % Keep track of the features that have been removed so we can use them
    % later
    nfeats = size(model.data.X,2);
    model.featsel.idx = 1:nfeats;

    disp('=========================================================================');
    disp('-> Calling SIFTED for auto-feature selection.');
    disp('=========================================================================');
    [model.data.X, model.sifted] = SIFTED(model.data.X, model.data.Y, model.data.Ybin, opts.sifted);
    model.data.featlabels = model.data.featlabels(model.sifted.selvars);
    model.featsel.idx = model.featsel.idx(model.sifted.selvars); 
    
    % -------------------------------------------------------------------------
    % Preparing the outputs for further analysis
    model.opts = opts;
    % -------------------------------------------------------------------------
    disp('-------------------------------------------------------------------------');
    disp('-> Storing the raw MATLAB results for post-processing and/or debugging.');
    save([rootdir 'model.mat'],'-struct','model'); % Save the main results
    save([rootdir 'workspace.mat']); % Save the full workspace for debugging

    % kept features
    sel_feats = strcat('feature_', model.data.featlabels);
    writeCell2CSV(sel_feats, compose('f%d', 1:length(sel_feats)), {'feature'}, strcat(rootdir, 'selected_features.csv'));
    
    % -------------------------------------------------------------------------
    disp(['-> Completed! Elapsed time: ' num2str(toc(startProcess)) 's']);
    disp('EOF:SUCCESS');

end

% =========================================================================
function [X, out] = SIFTED(X, Y, Ybin, opts)
    % -------------------------------------------------------------------------
    % SIFTED.m
    % -------------------------------------------------------------------------
    %
    % By: Mario Andres Munoz Acosta
    %     School of Mathematics and Statistics
    %     The University of Melbourne
    %     Australia
    %     2021
    %
    % -------------------------------------------------------------------------
    
    rng(111); 
    
    nfeats = size(X,2);
    if nfeats<=1
        error('-> There is only 1 feature. Stopping space construction.');
    elseif nfeats<=3
        disp('-> There are 3 or less features to do selection. Skipping feature selection.')
        out.selvars = 1:nfeats;
        return;
    end
    % ---------------------------------------------------------------------
    disp('-> Selecting features based on correlation with performance.');
    [out.rho,out.p] = corr(X,Y,'rows','pairwise');
    rho = out.rho;
    rho(isnan(rho) | (out.p>0.05)) = 0;
    [rho,row] = sort(abs(rho),1,'descend');
    out.selvars = false(1,nfeats);
    % Always take the most correlated feature for each algorithm
    out.selvars(unique(row(1,:))) = true;
    % Now take any feature that has correlation at least equal to opts.rho
    for ii=2:nfeats
        out.selvars(unique(row(ii,rho(ii,:)>=opts.rho))) = true;
    end
    out.selvars = find(out.selvars);
    Xaux = X(:,out.selvars);
    disp(['-> Keeping ' num2str(size(Xaux,2)) ' out of ' num2str(nfeats) ' features (correlation).']);
    % ---------------------------------------------------------------------
    nfeats = size(Xaux,2);
    if nfeats<=1
        error('-> There is only 1 feature. Stopping space construction.');
    elseif nfeats<=3
        disp('-> There are 3 or less features to do selection. Skipping correlation clustering selection.');
        X = Xaux;
        return;
    elseif nfeats<opts.K
        disp('-> There are less features than clusters. Skipping correlation clustering selection.');
        X = Xaux;
        return;
    end
    % ---------------------------------------------------------------------
    disp('-> Selecting features based on correlation clustering.');
    % nalgos = size(Ybin,2);
    state = rng;
    % rng('default');
    out.eva = evalclusters(Xaux', 'kmeans', 'Silhouette', 'KList', 3:nfeats, ... % minimum of three features
                                  'Distance', 'correlation');
    disp('-> Average silhouette values for each number of clusters.')
    disp([out.eva.InspectedK; out.eva.CriterionValues]);
    % ---------------------------------------------------------------------
    % if out.eva.CriterionValues(out.eva.InspectedK==opts.K)<0.5
    %     disp(['-> The silhouette value for K=' num2str(opts.K) ...
    %           ' is below 0.5. You should consider increasing K.']);
    %     out.Ksuggested = out.eva.InspectedK(find(out.eva.CriterionValues>0.75,1));
    %     if ~isempty(out.Ksuggested)
    %         disp(['-> A suggested value of K is ' num2str(out.Ksuggested)]);
    %     end
    % end
    
    %%%% DN ---------------------------------------------------------------------
    if opts.K==0
        out.localMaxK = out.eva.InspectedK(find(islocalmax(out.eva.CriterionValues),1));
        out.suggestedK = out.eva.InspectedK(find(out.eva.CriterionValues>0.7,1));
    
        disp('----------------------------------------------')
        disp(['K to try ---- ' num2str(out.suggestedK) ' , ' num2str(out.localMaxK)]);
    
        opts.K = out.suggestedK;
        [X_s, out_s] = findCombo(X, out, Ybin, opts, state);
    
        opts.K = out.localMaxK;
        [X_l, out_l] = findCombo(X, out, Ybin, opts, state);
        if out_s.besterr < out_l.besterr
            X = X_s;
            out = out_s;
        else
            X = X_l;
            out = out_l;
        end      
        
    elseif opts.K==-1
        % only local max K
        out.localMaxK = out.eva.InspectedK(find(islocalmax(out.eva.CriterionValues),1));
        opts.K = out.localMaxK;
        [X, out] = findCombo(X, out, Ybin, opts, state);
        
        
    else
        [X, out] = findCombo(X, out, Ybin, opts, state);        
    
    end
    
    disp(['-> Keeping ' num2str(size(X, 2)) ' out of ' num2str(nfeats) ' features (clustering).']);
    
end
% =========================================================================
function [X, out] = findCombo(X, out, Ybin, opts, state)

    % global gacostvals
    
    
    nworkers = 0;
    nalgos = size(Ybin,2);
    Xaux = X(:,out.selvars);
    nfeats = size(Xaux,2);
    
    % ---------------------------------------------------------------------
    % rng('default');
    out.clust = bsxfun(@eq, kmeans(Xaux', opts.K, 'Distance', 'correlation', ...
                                                'MaxIter', opts.MaxIter, ...
                                                'Replicates', opts.Replicates, ...
                                                'Options', statset('UseParallel', nworkers~=0), ...
                                                'OnlinePhase', 'on'), 1:opts.K);
    % rng(state);
    disp(['-> Constructing ' num2str(opts.K) ' clusters of features.']);
    % ---------------------------------------------------------------------
    % Using these out.clusters, determine all possible combinations that take one
    % feature from each out.cluster.
    strcmd = '[';
    for i=1:opts.K
        strcmd = [strcmd 'X' num2str(i) ]; %#ok<*AGROW>
        if i<opts.K
            strcmd = [strcmd ','];
        else
            strcmd = [strcmd '] = ndgrid('];
        end
    end
    for i=1:opts.K
        strcmd = [strcmd 'find(out.clust(:,' num2str(i) '))'];
        if i<opts.K
            strcmd = [strcmd ','];
        else
            strcmd = [strcmd ');'];
        end
    end
    eval(strcmd);
    strcmd = 'comb = [';
    for i=1:opts.K
        strcmd = [strcmd 'X' num2str(i) '(:)'];
        if i<opts.K
            strcmd = [strcmd ','];
        else
            strcmd = [strcmd '];'];
        end
    end
    eval(strcmd);
    
    ncomb = size(comb,1); %#ok<*NODEF>
    comb = sort(comb,2);
    disp(['-> ' num2str(ncomb) ' valid feature combinations.']);
    
    maxcomb = 100;
    % ---------------------------------------------------------------------
    % Determine which combination produces the best separation while using a
    % two dimensional PCA projection. The separation is defined by a Tree
    % Bagger.
    if ncomb>maxcomb
        disp(['-> There are over ' num2str(maxcomb) ' valid combinations. Using a GA+LookUpTable to find an optimal one.']);
        gacostvals = NaN.*ones(ncomb,1);
        fcnwrap = @(idx) fcnforga(comb,gacostvals,idx,Xaux,Ybin,opts.NTREES,out.clust,nworkers);
        gaopts = optimoptions('ga','FitnessLimit',0,'FunctionTolerance',1e-3,...
                            'MaxGenerations',100,'MaxStallGenerations',5,...
                            'PopulationSize',50); % This sets the maximum to 1000 combinations
        ind = ga(fcnwrap,opts.K,[],[],[],[],ones(1,opts.K),sum(out.clust),[],1:opts.K,gaopts);
        decoder = false(1,size(Xaux,2)); % Decode the chromosome
        for i=1:opts.K
            aux = find(out.clust(:,i));
            decoder(aux(ind(i))) = true;
        end
        out.selvars = out.selvars(decoder);
        %%%
        besterr = costfcn(decoder,Xaux,Ybin,opts.NTREES,nworkers);
        disp(['Average error:' num2str(mean(besterr,2))])
        out.besterr = besterr;
        %%%
        
    elseif ncomb==1
        disp('-> There is one valid combination. It will be considered the optimal one.');
        % out.selvars = 1:nfeats;
    else
        disp(['-> There are less than ' num2str(maxcomb) ' valid combinations. Using brute-force to find an optimal one.']);
        out.ooberr = zeros(ncomb,nalgos);
        for i=1:ncomb
            tic;
            out.ooberr(i,:) = costfcn(comb(i,:),Xaux,Ybin,opts.NTREES,nworkers);
            % etime = toc;
            % disp(['    -> Combination No. ' num2str(i) ' | Elapsed Time: ' num2str(etime,'%.2f\n') ...
            %       's | Average error : ' num2str(mean(out.ooberr(i,:)))]);
            tic;
        end
        [~,best] = min(sum(out.ooberr,2));
        disp(['Average error:' num2str(min(mean(out.ooberr,2)))]) %%%
        out.selvars = sort(out.selvars(comb(best,:)));
        out.besterr = min(mean(out.ooberr,2));
    end
    X = X(:,out.selvars);

end
% =========================================================================
function ooberr = costfcn(comb,X,Ybin,ntrees,nworkers)

    [~, score] = pca(X(:,comb), 'NumComponents', 2); %#ok<*IDISVAR> % Reduce using PCA
    nalgos = size(Ybin,2);
    ooberr = zeros(1,nalgos);
    for j = 1:nalgos
        % state = rng;
        % rng('default');
        tree = TreeBagger(ntrees, score, Ybin(:,j), 'OOBPrediction', 'on',...
                                                    'Options', statset('UseParallel', nworkers~=0));
        ooberr(j) = mean(Ybin(:,j)~=str2double(oobPredict(tree)));
        % rng(state);
    end    
end
% =========================================================================
function sumerr = fcnforga(comb,gacostvals,idx,X,Ybin,ntrees,clust,nworkers)
    % global gacostvals
    
    tic;
    % Decode the chromosome into a binary string representing the selected
    % features
    ccomb = false(1,size(X,2));
    for i=1:length(idx)
        aux = find(clust(:,i));
        ccomb(aux(idx(i))) = true;
    end
    % Calculate the cost function. Use the lookup table to reduce the amount of
    % computation.
    ind = find(all(comb==find(ccomb),2));
    if isnan(gacostvals(ind))
        ooberr = costfcn(ccomb,X,Ybin,ntrees,nworkers);
        sumerr = sum(ooberr);
        gacostvals(ind) = sumerr;
    else
        sumerr = gacostvals(ind);
    end
    
    % etime = toc;
    % disp(['    -> Combination No. ' num2str(ind) ' | Elapsed Time: ' num2str(etime,'%.2f\n') ...
    %       's | Average error : ' num2str(sumerr./size(Ybin,2))]);
    
end
% =========================================================================
