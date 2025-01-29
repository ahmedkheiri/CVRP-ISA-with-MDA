% ---- Adapted from 
% M.A. Muñoz and K. Smith-Miles. Instance Space Analysis: A toolkit for the assessment of algorithmic power. andremun/InstanceSpace on Github. Zenodo, DOI:10.5281/zenodo.4484107, 2020.
% ----

function out = TRACE(Z, Ybin, P, beta, algolabels, opts)

if size(Z,2) == 3 
    disp('  -> 3D Instance Space detected using TRACE2');
    opts.Trace2 = true;
end

if exist('gcp','file')==2
    mypool = gcp('nocreate');
    if ~isempty(mypool)
        nworkers = mypool.NumWorkers;
    else
        nworkers = 0;
    end
else
    nworkers = 0;
end
% -------------------------------------------------------------------------
% First step is to transform the data to the footprint space, and to
% calculate the 'space' footprint. This is also the maximum area possible
% for a footprint.
if opts.Trace2
    disp('  -> TRACE2 is calculating the space area and density.');
else
    disp('  -> TRACE is calculating the space area and density.');
end
ninst = size(Z,1);
nalgos = size(Ybin,2);
nclass = length(unique(P));

if opts.Trace2
    out.space = TRACEbuild2(Z, true(ninst,1), opts);
else
    out.space = TRACEbuild(Z, true(ninst,1), opts);
end
disp(['    -> Space area: ' num2str(out.space.area) ...
      ' | Space density: ' num2str(out.space.density)]);
% -------------------------------------------------------------------------
% This loop will calculate the footprints for good/bad instances and the
% best algorithm.
disp('-------------------------------------------------------------------------');
if opts.Trace2
    disp('  -> TRACE2 is calculating the algorithm footprints.');
else
    disp('  -> TRACE is calculating the algorithm footprints.');
end

good = cell(1,nalgos);
best = cell(1,nclass);
% Use the actual data to calculate the footprints
parfor (i=1:nalgos,nworkers)
    tic;
    disp(['    -> Good performance footprint for ''' algolabels{i} '''']);        
    if opts.Trace2
        good{i} = TRACEbuild2(Z, Ybin(:,i), opts);        
    else
        good{i} = TRACEbuild(Z, Ybin(:,i), opts);        
    end
    disp(['    -> Algorithm ''' algolabels{i} ''' completed. Elapsed time: ' num2str(toc,'%.2f\n') 's']);
end

parfor (i=1:nclass,nworkers)
    tic;
    if opts.Trace2
        best{i} = TRACEbuild2(Z, P==i, opts);
    else
        best{i} = TRACEbuild(Z, P==i, opts);
    end
end

out.good = good;
out.best = best;
out.bestPre = best;  %% to save the original best footprints

% -------------------------------------------------------------------------
% Detecting collisions and removing them. Original Trace ONLY
if ~opts.Trace2
    disp('-------------------------------------------------------------------------');
    disp('  -> TRACE is detecting and removing contradictory sections of the footprints.');
    for i=1:nclass
        disp(['  -> Base algorithm ''' algolabels{i} '''']);
        startBase = tic;
        for j=i+1:nclass 
            disp(['      -> TRACE is comparing ''' algolabels{i} ''' with ''' algolabels{j} '''']);
            startTest = tic;
            [out.best{i}, out.best{j}] = TRACEcontra(out.best{i}, out.best{j}, ...
                                                     Z, P==i, P==j, opts);%, false);

            disp(['      -> Test algorithm ''' algolabels{j} ...
                  ''' completed. Elapsed time: ' num2str(toc(startTest),'%.2f\n') 's']);
        end
        disp(['  -> Base algorithm ''' algolabels{i} ...
              ''' completed. Elapsed time: ' num2str(toc(startBase),'%.2f\n') 's']);
    end
end
% -------------------------------------------------------------------------
% Beta hard footprints. First step is to calculate them.
out.hard = TRACEbuild(Z, ~beta, opts);
% -------------------------------------------------------------------------
% Calculating performance
disp('-------------------------------------------------------------------------');
disp('  -> Preparing the summary table.');
out.summary = cell(nclass+1,11);
out.summary(1,2:end) = {'Area_Good',...
                        'Area_Good_Normalized',...
                        'Density_Good',...
                        'Density_Good_Normalized',...
                        'Purity_Good',...
                        'Area_Best',...
                        'Area_Best_Normalized',...
                        'Density_Best',...
                        'Density_Best_Normalized',...
                        'Purity_Best'};
out.summary(2:end,1) = algolabels;
for i=1:nalgos
    row = [TRACEsummary(out.good{i}, out.space.area, out.space.density), ...
           TRACEsummary(out.best{i}, out.space.area, out.space.density)];
    out.summary(i+1,2:end) = num2cell(round(row,3));
end
if nclass > nalgos
    for i=nalgos+1:nclass
        row = [zeros(1,5), ...
            TRACEsummary(out.best{i}, out.space.area, out.space.density)];
        out.summary(i+1,2:end) = num2cell(round(row,3));
    end
end
% disp(' ');
% disp(out.summary);

end
% =========================================================================
% SUBFUNCTIONS
% =========================================================================
function footprint = TRACEbuild(Z, Ybin, opts)
% check if Ybin is logical
if ~islogical(Ybin)
    Ybin = logical(Ybin);
end
% If there is no Y to work with, then there is not point on this one
Ig = unique(Z(Ybin,:),'rows');   % There might be points overlapped, so eliminate them to avoid problems
if size(Ig,1)<3
    footprint = TRACEthrow;
else
    footprint = struct;
end
    
nn = max(min(ceil(sum(Ybin)/20),50),3);
class = dbscan(Ig,nn); % Use DBSCAN to identify dense regions
flag = false;
for i=1:max(class) %Ignore -1/0
    polydata = Ig(class==i,:);
    polydata = polydata(boundary(polydata,1),:);
    aux = TRACEfitpoly(polydata,Z,Ybin, opts);
    if ~isempty(aux)
        if ~flag
            footprint.polygon = aux;
            flag = true;
        else
            footprint.polygon = union(footprint.polygon,aux);
        end
    end
end
if isfield(footprint,'polygon') && ~isempty(footprint.polygon)
    footprint.polygon = rmslivers(footprint.polygon,1e-2);
    footprint.area = area(footprint.polygon);
    footprint.elements = sum(isinterior(footprint.polygon,Z));
    footprint.goodElements = sum(isinterior(footprint.polygon,Z(Ybin,:)));
    footprint.density = footprint.elements./footprint.area;
    footprint.purity = footprint.goodElements./footprint.elements;
else
    footprint = TRACEthrow;
end

end
% =========================================================================
function footprint = TRACEbuild2(Z, Ybin, opts)

% If there is no Y to work with, then there is not point on this one
Ig = unique(Z(Ybin,:),'rows');   % There might be points overlapped, so eliminate them to avoid problems
if size(Ig,1)<3
    footprint = TRACEthrow;
else
    footprint = struct;
end
if ~isfield(opts,'prior')
    opts.prior = [0.6,0.4];
end
if ~isfield(opts,'nn')
    opts.nn=50;
end


if size(unique(Ybin),1) > 1
    knt = fitcknn(Z,Ybin,'Prior',opts.prior,'NumNeighbors',opts.nn); %Fit a KNN classification
    prt = predict(knt,Z); 
    polydata = Z(prt==1 & Ybin == 1,:); %Build poly data from instances correctly identfied as good from the KNN classifier
else
    %knt = fitcknn(Z,Ybin,'NumNeighbors',nn); %'Prior',[0.6,0.4],
    polydata = Z;
end
footprint.polygon = alphaShape(polydata); %Build the alpha shape from poly data
D = size(Z);


%Below removes outlier points untill the minimum purity threshold is met
if isfield(footprint,'polygon') && ~isempty(footprint.polygon.Points) && ~(footprint.polygon.Alpha==Inf) && D(2) == 2
    footprint.area = area(footprint.polygon);
    footprint.elements = sum(inShape(footprint.polygon,Z));
    footprint.goodElements = sum(inShape(footprint.polygon,Z(Ybin,:)));
    footprint.density = footprint.elements./footprint.area;
    footprint.purity = footprint.goodElements./footprint.elements;
    AS = alphaSpectrum(footprint.polygon);
    AS = footprint.polygon.Alpha:-((footprint.polygon.Alpha-min(AS))/100):min(AS);
    ii = 1;
    while footprint.purity < opts.PI && ii < size(AS,2)
        footprint.polygon.Alpha = AS(ii);
        footprint.area = area(footprint.polygon);
        footprint.polygon.RegionThreshold =  footprint.area/20; %footprint.polygon.RegionThreshold + *(1-footprint.purity)
        footprint.area = area(footprint.polygon);
        if footprint.area > 0
            footprint.elements = sum(inShape(footprint.polygon,Z));
            footprint.goodElements = sum(inShape(footprint.polygon,Z(Ybin,:)));
            footprint.purity = footprint.goodElements./footprint.elements;
            footprint.density = footprint.elements./footprint.area;
        else
            ii = size(AS,2);
            footprint = TRACEthrow;
        end
        ii = ii+1;
    end
elseif  isfield(footprint,'polygon') && ~isempty(footprint.polygon.Points) && ~(footprint.polygon.Alpha==Inf) && D(2) == 3
    footprint.area = volume(footprint.polygon);
    footprint.elements = sum(inShape(footprint.polygon,Z));
    footprint.goodElements = sum(inShape(footprint.polygon,Z(Ybin,:)));
    footprint.density = footprint.elements./footprint.area;
    footprint.purity = footprint.goodElements./footprint.elements;
    AS = alphaSpectrum(footprint.polygon);
    AS = footprint.polygon.Alpha:-((footprint.polygon.Alpha-min(AS))/100):min(AS);
    ii = 1;
    while footprint.purity < opts.PI && ii < size(AS,2)
        footprint.polygon.Alpha = AS(ii);
        footprint.area = volume(footprint.polygon);
        footprint.polygon.RegionThreshold =  footprint.area/20; %footprint.polygon.RegionThreshold + *(1-footprint.purity)
        footprint.area = volume(footprint.polygon);
        if footprint.area > 0
            footprint.elements = sum(inShape(footprint.polygon,Z));
            footprint.goodElements = sum(inShape(footprint.polygon,Z(Ybin,:)));
            footprint.purity = footprint.goodElements./footprint.elements;
            footprint.density = footprint.elements./footprint.area;
        else
            ii = size(AS,2);
            footprint = TRACEthrow;
        end
        ii = ii+1;
    end
    
    
else
    footprint = TRACEthrow;
end

end
% =========================================================================
function [base,test] = TRACEcontra(base,test,Z,Ybase,Ytest,opts)%,isbin)
% 
if isempty(base.polygon) || isempty(test.polygon)
    return;
end

maxtries = 3; % Tries once to tighten the bounds.
numtries = 1;
contradiction = intersect(base.polygon,test.polygon);
while contradiction.NumRegions~=0 && numtries<=maxtries
    numElements = sum(isinterior(contradiction,Z));
    numGoodElementsBase = sum(isinterior(contradiction,Z(Ybase,:)));
    numGoodElementsTest = sum(isinterior(contradiction,Z(Ytest,:)));
    purityBase = numGoodElementsBase/numElements;
    purityTest = numGoodElementsTest/numElements;
    
    if purityBase>purityTest %&& (~isbin || (purityBase>0.55 && isbin))
        carea = area(contradiction)./area(test.polygon);
        disp(['        -> ' num2str(round(100.*carea,1)) '%' ...
              ' of the test footprint is contradictory.']);
        test.polygon = subtract(test.polygon,contradiction);
        if numtries<maxtries
            test.polygon = TRACEtight(test.polygon,Z,Ytest,opts);
        end
    elseif purityTest>purityBase %&& (~isbin || (purityTest>0.55 && isbin))
        carea = area(contradiction)./area(base.polygon);
        disp(['        -> ' num2str(round(100.*carea,1)) '%' ...
              ' of the base footprint is contradictory.']);
        base.polygon = subtract(base.polygon,contradiction);
        if numtries<maxtries
            base.polygon = TRACEtight(base.polygon,Z,Ybase,opts);
        end
    else
        disp('        -> Purity of the contradicting areas is equal for both footprints.');
        disp('        -> Ignoring the contradicting area.');
        break;
    end
    if isempty(base.polygon) || isempty(test.polygon)
        break;
    else
        contradiction = intersect(base.polygon,test.polygon);
    end
    numtries = numtries+1;
end

if isempty(base.polygon)
    base = TRACEthrow;
else
    base.area = area(base.polygon);
    base.elements = sum(isinterior(base.polygon,Z));
    base.goodElements = sum(isinterior(base.polygon,Z(Ybase,:)));
    base.density = base.elements./base.area;
    base.purity = base.goodElements./base.elements;
end
if isempty(test.polygon)
    test = TRACEthrow;
else
    test.area = area(test.polygon);
    test.elements = sum(isinterior(test.polygon,Z));
    test.goodElements = sum(isinterior(test.polygon,Z(Ytest,:)));
    test.density = test.elements./test.area;
    test.purity = test.goodElements./test.elements;
end

end
% =========================================================================
function polygon = TRACEtight(polygon,Z,Ybin,opts)

splits = regions(polygon);
nregions = length(splits);
flags = true(1,nregions);

for i=1:nregions
    % Find the vertex of this polygon
    criteria = isinterior(splits(i),Z) & Ybin;
    % disp(criteria);
    polydata = Z(criteria,:);
    if size(polydata,1)<3
        flags(i) = false;
        continue
    end
    aux = TRACEfitpoly(polydata(boundary(polydata,1),:),Z,Ybin,opts);
    if isempty(aux)
        flags(i) = false;
        continue
    end
    splits(i) = aux;
end
if any(flags)
    polygon = union(splits(flags));
else
    polygon = [];
end

end
% =========================================================================
function polygon = TRACEfitpoly(polydata,Z,Ybin,opts)

warning('off','MATLAB:polyshape:repairedBySimplify');

if size(polydata,1)<3
    polygon = [];
    warning('on','MATLAB:polyshape:repairedBySimplify');
    return
end

polygon = polyshape(polydata,'Simplify',true);
polygon = rmslivers(polygon,5e-2);

if ~all(Ybin)
    if polygon.NumRegions<1
        polygon = [];
        warning('on','MATLAB:polyshape:repairedBySimplify');
        return
    end
    tri = triangulation(polygon);
    nrow = size(tri.ConnectivityList,1);
    for ii=1:nrow
        tridata = tri.Points(tri.ConnectivityList(ii,:),:);
        piece = polyshape(tridata,'Simplify',true);
        elements = sum(isinterior(piece,Z));
        goodElements = sum(isinterior(piece,Z(Ybin,:)));
        if opts.PI>(goodElements/elements)
            polygon = subtract(polygon,piece);
        end
    end
end

warning('on','MATLAB:polyshape:repairedBySimplify');

end
% =========================================================================
function out = TRACEsummary(footprint, spaceArea, spaceDensity)
% 
out = [footprint.area,...
       footprint.area/spaceArea,...
       footprint.density,...
       footprint.density/spaceDensity,...
       footprint.purity];
out(isnan(out)) = 0;

end
% =========================================================================
function footprint = TRACEthrow

disp('        -> There are not enough instances to calculate a footprint.');
disp('        -> The subset of instances used is too small.');
footprint.polygon = [];
footprint.area = 0;
footprint.elements = 0;
footprint.goodElements = 0;
footprint.density = 0;
footprint.purity = 0;

end
% =========================================================================
% Function: [class,type]=dbscan(x,k,Eps)
% -------------------------------------------------------------------------
% Aim: 
% Clustering the data with Density-Based Scan Algorithm with Noise (DBSCAN)
% -------------------------------------------------------------------------
% Input: 
% x - data set (m,n); m-objects, n-variables
% k - number of objects in a neighborhood of an object 
% (minimal number of objects considered as a cluster)
% Eps - neighborhood radius, if not known avoid this parameter or put []
% -------------------------------------------------------------------------
% Output: 
% class - vector specifying assignment of the i-th object to certain 
% cluster (m,1)
% type - vector specifying type of the i-th object 
% (core: 1, border: 0, outlier: -1)
% -------------------------------------------------------------------------
% Example of use:
% x=[randn(30,2)*.4;randn(40,2)*.5+ones(40,1)*[4 4]];
% [class,type]=dbscan(x,5,[]);
% -------------------------------------------------------------------------
% References:
% [1] M. Ester, H. Kriegel, J. Sander, X. Xu, A density-based algorithm for 
% discovering clusters in large spatial databases with noise, proc. 
% 2nd Int. Conf. on Knowledge Discovery and Data Mining, Portland, OR, 1996, 
% p. 226, available from: 
% www.dbs.informatik.uni-muenchen.de/cgi-bin/papers?query=--CO
% [2] M. Daszykowski, B. Walczak, D. L. Massart, Looking for 
% Natural Patterns in Data. Part 1: Density Based Approach, 
% Chemom. Intell. Lab. Syst. 56 (2001) 83-92 
% -------------------------------------------------------------------------
% Written by Michal Daszykowski
% Department of Chemometrics, Institute of Chemistry, 
% The University of Silesia
% December 2004
% http://www.chemometria.us.edu.pl

function [class,type]=dbscan(x,k,Eps)

m=size(x,1);

if nargin<3 || isempty(Eps)
   [Eps]=epsilon(x,k);
end

x=[(1:m)' x];
[m,n]=size(x);
type=zeros(1,m);
no=1;
touched=zeros(m,1);
class=zeros(1,m);
for i=1:m
    if touched(i)==0
       ob=x(i,:);
       D=dist(ob(2:n),x(:,2:n));
       ind=find(D<=Eps);
    
       if length(ind)>1 && length(ind)<k+1       
          type(i)=0;
          class(i)=0;
       end
       if length(ind)==1
          type(i)=-1;
          class(i)=-1;  
          touched(i)=1;
       end

       if length(ind)>=k+1
          type(i)=1;
          class(ind)=ones(length(ind),1)*max(no);
          
          while ~isempty(ind)
                ob=x(ind(1),:);
                touched(ind(1))=1;
                ind(1)=[];
                D=dist(ob(2:n),x(:,2:n));
                i1=find(D<=Eps);
     
                if length(i1)>1
                   class(i1)=no;
                   if length(i1)>=k+1
                      type(ob(1))=1;
                   else
                      type(ob(1))=0;
                   end

                    for j=1:length(i1)
                        if touched(i1(j))==0
                            touched(i1(j))=1;
                            ind=[ind i1(j)];
                            class(i1(j))=no;
                        end
                    end
                end
          end
          no=no+1; 
       end
   end
end

i1=find(class==0);
class(i1)=-1;
type(i1)=-1;

end
% =========================================================================
function [Eps]=epsilon(x,k)

% Function: [Eps]=epsilon(x,k)
%
% Aim: 
% Analytical way of estimating neighborhood radius for DBSCAN
%
% Input: 
% x - data matrix (m,n); m-objects, n-variables
% k - number of objects in a neighborhood of an object
% (minimal number of objects considered as a cluster)

[m,n]=size(x);

Eps=((prod(max(x)-min(x))*k*gamma(.5*n+1))/(m*sqrt(pi.^n))).^(1/n);

end
% =========================================================================
function [D]=dist(i,x)

% function: [D]=dist(i,x)
%
% Aim: 
% Calculates the Euclidean distances between the i-th object and all objects in x	 
%								    
% Input: 
% i - an object (1,n)
% x - data matrix (m,n); m-objects, n-variables	    
%                                                                 
% Output: 
% D - Euclidean distance (m,1)

[m,n]=size(x);
D=sqrt(sum((((ones(m,1)*i)-x).^2)'));

if n==1
   D=abs((ones(m,1)*i-x))';
end

end
% =========================================================================
