% ---- Adapted from 
% M.A. Muñoz and K. Smith-Miles. Instance Space Analysis: A toolkit for the assessment of algorithmic power. andremun/InstanceSpace on Github. Zenodo, DOI:10.5281/zenodo.4484107, 2020.
% ----

function model = TRACEtest(model, Z, Ybin, P, beta, algolabels)

    nclass = length(model.best);
    nalgos = length(model.good);

    disp('-------------------------------------------------------------------------');
    disp('  -> TRACE is calculating the algorithm footprints.');
    model.test.best = zeros(nclass,5);
    model.test.good = zeros(nalgos,5);
    % model.test.bad = zeros(nalgos,5);
    % Use the actual data to calculate the footprints
    for i=1:nalgos
        model.test.good(i,:) = TRACEtestsummary(model.good{i}, Z,  Ybin(:,i), model.space.area, model.space.density);
    end
    for i=1:nclass
        model.test.best(i,:) = TRACEtestsummary(model.best{i}, Z,  P==i, model.space.area, model.space.density);       
    end
    
    % -------------------------------------------------------------------------
    % Beta hard footprints. First step is to calculate them.
    % model.test.easy = TRACEtestsummary(model.easy, Z,  beta, model.space.area, model.space.density);
    model.test.hard = TRACEtestsummary(model.hard, Z, ~beta, model.space.area, model.space.density);
    % -------------------------------------------------------------------------
    % Calculating performance
    disp('-------------------------------------------------------------------------');
    disp('  -> TRACE test is preparing the summary table.');
    model.summary = cell(nclass+1,11);
    model.summary(1,2:end) = {'Area_Good',...
                              'Area_Good_Normalized',...
                              'Density_Good',...
                              'Density_Good_Normalized',...
                              'Purity_Good',...
                              'Area_Best',...
                              'Area_Best_Normalized',...
                              'Density_Best',...
                              'Density_Best_Normalized',...
                              'Purity_Best'};
    model.summary(2:end,1) = algolabels;
    % model.summary(2:end,2:end) = num2cell([model.test.good model.test.best]);
    for i=1:nalgos
        row = [model.test.good(i,:), model.test.best(i,:)];
        % model.summary{i+1,2:end} = num2cell(row);
        model.summary(i+1, 2:end) = num2cell(row);

    end
    if nclass > nalgos
        for i=nalgos+1:nclass
            model.summary(i+1,2:end) = num2cell([zeros(1,5), model.test.best(i,:)]);
        end
    end
    
    % disp('  -> TRACE has completed. Footprint analysis results:');
    % disp(' ');
    % disp(model.summary);
    
    end
    % =========================================================================
    function out = TRACEtestsummary(footprint, Z, Ybin, spaceArea, spaceDensity)
    % 
    if isempty(footprint.polygon) || all(~Ybin)
        out = zeros(5,1);
    else
        elements = sum(isinterior(footprint.polygon, Z));
        goodElements = sum(isinterior(footprint.polygon, Z(Ybin,:)));
        density = elements./footprint.area;
        purity = goodElements./elements;
        
        out = [footprint.area,...
               footprint.area/spaceArea,...
               density,...
               density/spaceDensity,...
               purity];
    end
    out(isnan(out)) = 0;
    end
    % =========================================================================
