
function [Network] = Backprop(Network,image,class,learningRate)


% Backwards Operation
BackpropNetwork = Network(1);

for Layer = size(Network,2):-1:1

    CurrentLayer = Network(Layer);
    BackpropNetwork(Layer).Out =[];

    switch CurrentLayer.Name

        case 'Input'

            % Do Nothing

        case 'Conv'

            % Move local gradients to maxpool layer
            max_backprop = reshape(BackpropNetwork(Layer).In,[size(Network(Layer).Out)]);

            %Max Pooling to Cnv
            convBack = MaxPoolB(Network(Layer).params.Convolved,max_backprop,Network(Layer).params.Poolidx,Network(Layer).weights);

            %Rotate & Convolved filter with input image again

            [convolved] = Convolve(image,rot90(convBack,2));

            % f(n+1) = f + n*deltaFilt
            Network(Layer).weights = Network(Layer).weights + (learningRate*convolved);

            BackpropNetwork(Layer).Out = convolved;
            BackpropNetwork(Layer-1).In = BackpropNetwork(Layer).Out;

        case 'FC'

            % apply formula for the input layer
            BackpropNetwork(Layer).Out = Network(Layer).weights(:,1:end-1)' * BackpropNetwork(Layer).In;
            BackpropNetwork(Layer-1).In = BackpropNetwork(Layer).Out;


            flatlayer = reshape(Network(Layer).In, [], 1);
            F = [flatlayer;Network(Layer).bias];


            if any(~isfinite(Network(Layer).weights))
                k = 0;
            end

            if any(~isfinite(learningRate*(BackpropNetwork(Layer).In*F')))
               k = 0;
            end

            Network(Layer).weights = Network(Layer).weights+(learningRate*(BackpropNetwork(Layer).In*F'));



        case 'Output'

            BackpropNetwork(Layer).In = [Network(Layer).Out];

            
            % To make simple, we accept S as the sum of all exp(v)

            % This is the Partial derivative of the error function
            %https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy

            %CrossEntropy / LLE
            %o1 = Network(Layer).params.Softmax(class);
            %o2 = Network(Layer).params.Softmax(~class);

            %1, find the derivative of error respect to o1 and o2
            % that will be dL/do1 = -t1 / o1
            % that will be dL/do2 = -t2 / o2

            %2, find the derivative of oj respect to y1
            % for o1, that should be o1(1 - o1) 
            %do1dy1 = o1*(1 - o1);
            % for o2, that should be -o2*o1
            %do2dy1 = -1*o2*o1;

            %3, dy1 to it's own weights is h2
            %Complete chain rule to be h2 (o1 - t1),

            % we reserve the h2 multiplication for the next layer
            %Souts = Network(Layer).params.Softmax - class';

            S = sum(exp(Network(Layer).In));
            tj = -1/(Network(Layer).params.Softmax(class));
            o1 = ([exp(Network(Layer).In(class))*(S-exp(Network(Layer).In(class)))] / S^2);
            o2 = (-prod(exp(Network(Layer).In)) / S^2);
            Souts(class) = -tj*o1;
            Souts(~class) = -tj*o2;

            if any(~isfinite(Souts))
                k = 0;
            end

            BackpropNetwork(Layer).Out = Souts';
            BackpropNetwork(Layer-1).In = Souts';

    end

end

end