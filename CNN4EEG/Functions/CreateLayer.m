function Layer = CreateLayer(Layer,LayerType,varargin)

NewLayer.Name = LayerType;
NewLayer.inputSize = [];
NewLayer.bias = 1;
NewLayer.weights = [];
NewLayer.outputSize = [];
NewLayer.params = [];
NewLayer.In = [];
NewLayer.Out = [];

switch LayerType

    case 'Input'
        NewLayer.inputSize = varargin{1};
        NewLayer.outputSize = varargin{1};

        if ~isempty(Layer)
            error('Currrently supports input layer as first layer only')
        else
            Layer = NewLayer;
            Layer = Layer(false);
        end

    case 'Conv'
         PreviousLayer = Layer(end);

         NewLayer.inputSize = PreviousLayer.outputSize;
         NewLayer.params.FiltSize = varargin{1};
         NewLayer.weights = [rand([varargin{1}])*2]-1;

         NewLayer.weights = NewLayer.weights/10;

         NewLayer.params.StrideSize =  varargin{2};

         FiltSize = varargin{1}(2:end);
         ConvSize = [varargin{1}(1) NewLayer.inputSize-[FiltSize-1]];
         StrideSize = varargin{2};
         PoolSize = [[ConvSize - StrideSize]./StrideSize] + 1;

         NewLayer.outputSize = PoolSize;

    case 'FC'

         PreviousLayer = Layer(end);
         NewLayer.inputSize = PreviousLayer.outputSize;
         NewLayer.outputSize = varargin{1};
         NewLayer.weights = [rand([varargin{1} prod(NewLayer.inputSize)+1])*2]-1;

         NewLayer.weights = NewLayer.weights/10;

    case 'Output'
         PreviousLayer = Layer(end);
         NewLayer.inputSize = PreviousLayer.outputSize;
         NewLayer.outputSize = varargin{1};

end

Layer(end+1) = NewLayer;

end