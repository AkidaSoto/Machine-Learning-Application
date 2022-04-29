function [convolved] = Convolve(image,Filters)

%To do 
% Make sure the convolvution works in 5D
% Create Neighbor convolution for channel dimension


% define the parameters
padding = 0;
strideConv = 1;
% get the size of image and filter
[rowsI,colsI] = size(image);
[FiltF,rowsF,colsF] = size(Filters);
%Calculate the size of convolved img so we can make an empty array with
%these sizes
convolvedRow = ((rowsI+2*padding-rowsF)/strideConv)+1;
convolvedCol = ((colsI+2*padding-colsF)/strideConv)+1;

convolved = zeros(FiltF,convolvedRow, convolvedCol);
% index for go through rows and columns
indexr = rowsI-(rowsF)+1;
indexc = colsI-(colsF)+1;

for filt = 1:FiltF
    for row = 1:indexr
        for col = 1:indexc
            window = image(row:row+(rowsF-1), col:col+(colsF-1)); % Convolution Window
            multiple = window.*permute(Filters(filt,:,:),[2 3 1]); %element-wise multiplication of filter
            theSum = sum(multiple, 'all'); %sum of all elements within window
            convolved(filt,row,col) = theSum;
        end
    end
end
end