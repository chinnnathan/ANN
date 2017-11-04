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
        private double _prediction = 0;
        private double _bpropValue = 0;
        private int _momentum = 0;
        private List<NeuronInterconnect> _output = new List<NeuronInterconnect>();
        private List<NeuronInterconnect> _input = new List<NeuronInterconnect>();
        private List<double> _weights;
        //private Func<double, double> _activation;
        //private Func<double, double> _gradient;
        private Delegate _activation;
        private Delegate _gradient;

        //interconnects for neuron
        public List<NeuronInterconnect> InterOut { get { return _output; } set { _output = value; } }
        public List<NeuronInterconnect> InterIn { get { return _input; } set { _input = value; } }

        //neuron values for output files
        public string Name { get; set; }
        public double Input { get; set; }
        public double Prediction { get { return _prediction; } set { _prediction = value; } }
        public double BpropValue { get { return _bpropValue; } set { _bpropValue = value; } }
        public int Momentum { get { return _momentum; } set { _momentum = value; } }

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
}
