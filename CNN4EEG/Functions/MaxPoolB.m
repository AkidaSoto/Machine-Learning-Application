function [convBack] = poolToCnv2(convolved,max_backprop,maxLayeridx,Filters)

% ASSUME THAT POOLING STRIDE IS EQUAL TO FILTER SIZE
striderow = size(Filters,2);
stridecol = size(Filters,3);
filterSize =  size(Filters,2);
FiltF = size(Filters,1);

[rowsI,colsI,layerI] = size(convolved);
pooledrow = ((rowsI-striderow)/striderow)+1;
pooledcol = ((colsI-stridecol)/stridecol)+1;

% operations for maxpooling C layer using window size 2*2 and stride 2
% 3rd dimension is double the layers of filters, because after every
% layer, the second layer stores the indexes, used for backpropagation
C = zeros(rowsI, colsI,layerI); % maxpool layer
indexr = (pooledrow*striderow)-(striderow-1);
indexc = (pooledcol*stridecol)-(stridecol-1);

nfilt = 1:FiltF;
nrow = 1:2:indexr;
ncol = 1:2:indexc;
for f = 1:length(nfilt)
    for r = 1:length(nrow)
        for c = 1:length(ncol)

            indx = maxLayeridx(r,c,f);%index of
            value = max_backprop(r,c,f);

            window = C(nrow(r):nrow(r)+(filterSize-1), ncol(c):ncol(c)+(filterSize-1),f);
            window(indx) = value;
            C(nrow(r):nrow(r)+(filterSize-1), ncol(c):ncol(c)+(filterSize-1),f) = window;
        end
    end
end
% apply ReLU for backprob to.
C(C < 0) = 0;
convBack = C;

end