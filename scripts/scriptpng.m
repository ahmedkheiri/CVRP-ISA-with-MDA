% ---- Adapted from 
% M.A. Muñoz and K. Smith-Miles. Instance Space Analysis: A toolkit for the assessment of algorithmic power. andremun/InstanceSpace on Github. Zenodo, DOI:10.5281/zenodo.4484107, 2020.
% ----

function scriptpng(Z,Ybin, P, trace, algolabels, proj, rootdir)
    
    % -------------------------------------------------------------------------
    % Preliminaries
    scriptfcn;
    colormap('parula');
    nalgos = size(Ybin,2);
    Yfoot = Ybin;
    Pfoot = P;
    % -------------------------------------------------------------------------
    disp('=========================================================================');
    disp('-> Producing the plots.');
    
    % -------------------------------------------------------------------------
    % Drawing algorithm performance/footprint plots
    for i=1:nalgos
        try 
            clf;
            drawGoodBadFootprint(Z, ...
                                 trace.good{i}, ...
                                 Yfoot(:,i), ...
                                 strrep(algolabels{i},'_',' '));
            print(gcf,'-dpng',[rootdir proj 'footprint_' algolabels{i} '.png']);
        catch
            disp('No Footprint has been calculated');
        end
    end
    % Drawing the footprints as portfolio.
    clf;
    drawPortfolioFootprint(Z, trace.best, Pfoot, algolabels);
    print(gcf,'-dpng',[rootdir proj 'footprint_portfolio.png']);
    
