function [RMSE,Acc] = PredictModel(Network,XTest,YTest)
   
    % This function is for calculating accuracy 
    % 
    
    % parameters and values for accuracy calculations
    
    for i = 1:size(XTest,5)
        image = XTest(:,:,:,:,i);
        class = YTest(i,1) == unique(YTest);

        % Forward and get output
        Network = ForwardModel(Network,image,class);

        RMSE(i,:) = Network(end).params.Softmax - class';

        Acc(i) = all(Network(end).params.Softmax(class) > Network(end).params.Softmax(~class));
    end
    
    Acc = mean(Acc,2);

    RMSE = sqrt(sum(RMSE.^2,1)/size(XTest,5));

    

end