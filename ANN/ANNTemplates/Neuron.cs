using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.ANNTemplates
{
    public class Neuron
    {
        //internals for neuron
        private double _prediction;
        private double[] _predictions = new double[2];
        private double _bpropValue = 0;
        private int _momentum = 0;
        private List<NeuronInterconnect> _output = new List<NeuronInterconnect>();
        private List<NeuronInterconnect> _input = new List<NeuronInterconnect>();
        private List<double> _weights;
        private double _expected;
        private double _error;
        private double _lr = 0.01;
        //private Func<double, double> _activation;
        //private Func<double, double> _gradient;
        private Delegate _activation;
        private Delegate _gradient;

        //neuron values for output files
        public string Name { get; set; }
        public double LearningRate { get { return _lr; } set { _lr = value; } }
        public double BpropValue { get { return _bpropValue; } set { _bpropValue = value; } }
        public int Momentum { get { return _momentum; } set { _momentum = value; } }
        public double Expected { get { return _expected; } set { _expected = value; } }
        public double Prediction { get { return _prediction; } set { _prediction = value; } }
        public double[] Predictions { get { return _predictions; } set { _predictions = value; } }
        public double Error { get { return _error; } set { _error = value; } }
        public List<double> Weights { get { return _weights; } set { _weights = value; } }
        public double Input { get; set; }
        public double Update { get; set; }

        //interconnects for neuron
        public List<NeuronInterconnect> InterOut { get { return _output; } set { _output = value; } }
        public List<NeuronInterconnect> InterIn { get { return _input; } set { _input = value; } }

        public List<Neuron> PriorNeurons { get; set; } //neurons connected closer to input
        public List<Neuron> AfterNeurons { get; set; } //neurons connected closer to output

        public Neuron()
        {
            PriorNeurons = new List<Neuron>();
            AfterNeurons = new List<Neuron>();
        }

        //public Neuron(Func<double, double> activation, Func<double, double> gradient)
        public Neuron(Delegate activation, Delegate gradient)
        {
            _activation = activation;
            _gradient = gradient; 
            PriorNeurons = new List<Neuron>();
            AfterNeurons = new List<Neuron>();
        }

        public object RunActivation(params object[] args)
        {
            return _activation.DynamicInvoke(args);
        }

        public object RunGradient(params object[] args)
        {
            return _gradient.DynamicInvoke(args);
        }

    }

    public class DynamicNeuron
        ///<summary> The Dynamic Neuron is for the Clustering Algorithms </summary>
    {
        private int _dimensions = 1;
        private bool _ignore = true;
        private List<DynamicInterconnect> _inInter;
        private List<DynamicInterconnect> _outInter;
        private List<DynamicInterconnect> _sideInter;
        private List<DynamicNeuron> _inNeurons;
        private List<DynamicNeuron> _outNeurons;
        private List<DynamicNeuron> _sideNeurons;
        private List<Tuple<DynamicNeuron, DynamicInterconnect>> _input;
        private List<Tuple<DynamicNeuron, DynamicInterconnect>> _output;
        private List<Tuple<DynamicNeuron, DynamicInterconnect>> _side;
        private NeuronInternal _internal;

        public NeuronInternal Internal { get { return _internal; } set { _internal = value; } }
        public double[] Update { get { return _internal.Update; } set { _internal.Update = value; } }
        public int Index { get { return _internal.Index; } set { _internal.Index = value; } }
        public List<double> Weights { get { return _internal.Weights; } set { _internal.Weights = value; } }
        public List<DynamicNeuron> OutNeuron { get { return _outNeurons; } set { _outNeurons = value; } }
        public List<DynamicNeuron> InNeuron { get { return _inNeurons; } set { _inNeurons = value; } }
        public List<DynamicNeuron> SideNeuron { get { return _sideNeurons; } set { _sideNeurons = value; } }
        public List<DynamicInterconnect> OutInter { get { return _outInter; } set { _outInter = value; } }
        public List<DynamicInterconnect> InInter { get { return _inInter; } set { _inInter = value; } }
        public List<DynamicInterconnect> SideInter { get { return _sideInter; } set { _sideInter = value; } }
        public bool Ignore { get { return _ignore; } set { _ignore = value; } }

        public List<Tuple<DynamicNeuron, DynamicInterconnect>> Output
        {
            get { return _output; }

            set
            {
                _output = new List<Tuple<DynamicNeuron, DynamicInterconnect>>();
                var zval = _outNeurons.Zip(_outInter, (x, y) => new { N = x, I = y });
                foreach (var ni in zval)
                {
                    _output.Add(new Tuple<DynamicNeuron, DynamicInterconnect>(ni.N, ni.I));
                }
            }
        }

        public List<Tuple<DynamicNeuron, DynamicInterconnect>> Input
        {
            get { return _input; }

            set
            {
                _output = new List<Tuple<DynamicNeuron, DynamicInterconnect>>();
                var zval = _inNeurons.Zip(_inInter, (x, y) => new { N = x, I = y });
                foreach (var ni in zval)
                {
                    _output.Add(new Tuple<DynamicNeuron, DynamicInterconnect>(ni.N, ni.I));
                }
            }
        }

        public List<Tuple<DynamicNeuron, DynamicInterconnect>> Side
        {
            get { return _side; }

            set
            {
                value = new List<Tuple<DynamicNeuron, DynamicInterconnect>>();
                var zval = _sideNeurons.Zip(_sideInter, (x, y) => new { N = x, I = y });
                foreach (var ni in zval)
                {
                    value.Add(new Tuple<DynamicNeuron, DynamicInterconnect>(ni.N, ni.I));
                }
            }
        }

    }

}
