function [Network] = ForwardModel(Network,image,class)

% Forward Operation

for Layer = 1:size(Network,2)

    CurrentLayer = Network(Layer);

    switch CurrentLayer.Name

        case 'Input'

            Network(Layer).In = image;
            Network(Layer).Out = image;
            Network(Layer+1).In = image;

        case 'Conv'

            % Convolution
            [convolved] = Convolve(Network(Layer).In,Network(Layer).weights);

            % Relu
            convolved(convolved < 0) = 0;

            Network(Layer).params.Convolved = convolved;

            %Max Pool Layer
            [maxLayer,maxLayeridx] = MaxPool(convolved,Network(Layer));

            if any(~isfinite(maxLayer))
                k = 0;
            end

            Network(Layer).Out = maxLayer;
            Network(Layer).params.Poolidx = maxLayeridx;
            Network(Layer+1).In = maxLayer;

        case 'FC'

            %Flatten layer
            flatlayer = reshape(Network(Layer).In, [], 1);
            F = [flatlayer;Network(Layer).bias];

            Network(Layer).params.F = F;

            %Fully Connected layer
            Vout= F' * Network(Layer).weights';

            if any(~isfinite(Vout))
                k = 0;
            end

            %sigmoid activation function
            %Network(Layer).Out = 1.0 ./ ( 1.0 + exp(-Vout));
            %Network(Layer).Out = Vout/1000;

            %normalization
            Network(Layer).Out = Vout - max(Vout);
            Network(Layer+1).In = Network(Layer).Out;


        case 'Output'

            % Softmax Classification
            Network(Layer).params.Softmax = arrayfun(@(c) exp(c)/[sum(exp(Network(Layer).In))], Network(Layer).In);
            
            Network(Layer).params.Softmax(Network(Layer).params.Softmax < 1.0000e-15) = 1.0000e-15;

            % The Cross-Entropy
            Network(Layer).Out= -sum(class'.* log(Network(Layer).params.Softmax));
            
    end

end


end