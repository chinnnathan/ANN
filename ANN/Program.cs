using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using ANN.Utils;
using ANN.Networks;
using ANN.ANNTemplates;

namespace ANN
{
    public class ANN
    {
        private string  _networkType = "Backpropagation";
        private string  _trainingFile = "ASDNT0043_Site_1_ZEP0001GU5_TEST.csv";
        private string  _testingFile = "ASDNT0043_Site_1_ZEP0001GU5_TEST.csv";
        private string  _validationFile = "ASDNT0043_Site_1_ZEP0001GU5_VALID.csv";
        private string  _outputTestingFile = "TestingOutput.csv";
        private string  _outputTrainingFile = "TrainingOutput.csv";
        private double  _learningRate = 0.005;
        private int     _hiddenLayers = 1;
        private int     _classes = 7;
        private int     _cluster = 1;
        private int     _diameter = 2;
        private int[]   _nodes = new int[] { 15, 25, 35, 45, 55};
        private int     _epochs = 50;
        private bool    _normalize = false;
        private bool    _newNetwork = true;
        private bool    _train = true;
        private bool    _run = false;
        private bool    _debug = true;
        private dynamic _network; //have to abstract for multiple networks 
        private delegate double Del(double input);

        public string NetworkType { get { return _networkType; } set { _networkType=value; } }
        public string TestingFile { get { return _testingFile; } set { _testingFile=value; } }
        public string TrainingFile { get { return _trainingFile; } set { _trainingFile=value; } }
        public string ValidationFile { get { return _validationFile; } set { _validationFile=value; } }
        public string OutputTestingFile { get { return _outputTestingFile; } set { _outputTestingFile=value; } }
        public string OutputTrainingFile { get { return _outputTrainingFile; } set { _outputTrainingFile=value; } }
        public double LearningRate { get { return _learningRate; } set { _learningRate=value; } }
        public int HiddenLayers { get { return _hiddenLayers; } set { _hiddenLayers = value; } }
        public int Classes { get { return _classes; } set { _classes = value; } }
        public int Cluster { get { return _cluster; } set { _cluster = value; } }
        public int Diameter { get { return _diameter; } set { _diameter = value; } }
        public int Epochs { get { return _epochs; } set { _epochs = value; } }
        public int[] Nodes { get { return _nodes; } set { _nodes = value; } }
        public bool Normalize { get { return _normalize; } set { _normalize = value; } }
        public bool NewNetwork { get { return _newNetwork; } set { _newNetwork = value; } }
        public bool Train { get { return _train; } set { _train = value; } }
        public bool Run { get { return _run; } set { _run = value; } }
        public bool Debug { get { return _debug; } set { _debug = value; } }

        static void Main(string[] args)
        {
            ANN pr = new ANN();
            pr.RunCLI(args);
        }


        public void RunCLI(string[] args)
        {
            string ui = "r";
            do
            {
                RunProgram(args);
                Console.WriteLine("Enter q to quit: ");
                ui = Console.ReadLine();
                if (ui == "q")
                    break;
                else
                    args = ui.Split(' ').ToList().ToArray();
            } while (ui != "q");

        }


        public void RunProgram(string[] args)
        {
            if (args.Any(x=>x.ToLower()=="-help"))
            {
                this.PrintJSONString();
                return;
            }
            SetVariables(args);
            InterpretVariables();
        }

        private void SetVariables(string[] args)
        {
            for (int i=0; i<args.Length; i+=2)
            {
                string var = args[i].Trim('-');
                string value = args[i + 1];
                this.SetValueByString(var, value);
            }
        }

        private bool _networkDefined = false;

        private void InterpretVariables()
        {
            int nodes = 7;
            Del _activation = ActivationFunctions.Tanh;
            Del _gradient = GradientFunctions.Tanh;
            bool initneeded = !_networkDefined || _newNetwork;

            if (initneeded)
            {
                if (NetworkType.ToLower() == "backpropagation")
                {
                    _networkDefined = true;
                    _network = new Backpropagation(_classes, _hiddenLayers)
                    {
                        Debug = _debug,
                    };
                }

                if (NetworkType.ToLower() == "recurrent")
                {
                    _networkDefined = true;
                    _network = new Recurrent(_classes, _hiddenLayers)
                    {
                        Debug = _debug,
                    };
                }

                if (NetworkType.ToLower() == "som")
                {
                    _networkDefined = true;
                    _network = new SOM(_classes, _hiddenLayers)
                    {
                        Debug = _debug,
                    };
                    _network.Diameter = Diameter;
                }

                _network.Classes = _classes;
               
            }

            _network.Epochs = _epochs;
            _network.Debug = _debug;
            _network.SetInputs(_trainingFile, _cluster, _normalize);
            _network.SetCorrect(_validationFile, _classes);

            if (initneeded)
                _network.InitializeStd(_nodes, _activation, _gradient);


            if (_networkDefined)
            {
                bool nwt = (_train) ? _network.Train(Debug) : false;
                if (nwt)
                {
                    Console.WriteLine("Network: {0} Trained Succesfully", NetworkType);
                }
                else
                {
                    Console.WriteLine("Network: {0} Failed Training", NetworkType);
                }

                bool nwr = (_run) ? _network.Run() : false;
                if (nwr)
                {
                    Console.WriteLine("Network: {0} Ran Succesfully", NetworkType);
                }
                else
                {
                    Console.WriteLine("Network: {0} Failed", NetworkType);
                }
            }
        }
    }
}
