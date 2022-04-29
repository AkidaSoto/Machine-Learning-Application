function [maxLayer,maxLayeridx] = maxPool2(convolved,Layer)

ConvSize = size(convolved);
%Max Pooling
StrideSize = Layer.params.StrideSize;

% operations for maxpooling C layer using window size 2*2 and stride 2
% 3rd dimension is double the layers of filters, because after every
% layer, the second layer stores the indexes, used for backpropagation
maxLayer = zeros(Layer.outputSize); % maxpool layer
maxLayeridx = zeros(Layer.outputSize); % maxpool layer

nfilt = 1:StrideSize(1):ConvSize(1);
nrow = 1:StrideSize(2):ConvSize(2);
ncol = 1:StrideSize(3):ConvSize(3);

for f = 1:length(nfilt)
    for r = 1:length(nrow)
        for c = 1:length(ncol)
            window = convolved(nfilt(f):nfilt(f)+(StrideSize(1)-1),...
                               nrow(r):nrow(r)+(StrideSize(2)-1), ...
                               ncol(c):ncol(c)+(StrideSize(3)-1)); % sliding window
            
            [maxElement, maxIdx] = max(window, [], 'all', 'linear'); % find max of window with idx
            maxLayer(f,r, c) = maxElement; % set max elements on first 'page'
            maxLayeridx(f,r,c) = maxIdx; % set max element idx on second 'page'
        end
    end
end