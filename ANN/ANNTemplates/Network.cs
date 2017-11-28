using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Text.RegularExpressions;
using ANN.Utils;

namespace ANN.ANNTemplates
{
    public class Network : List<Layer>
    {
        protected double _incorrectTrain = 0.01;
        protected double _correctTrain = 0.99;
        protected double _barrierTrain = 0.5;
        private bool _debug = true;
        private bool _finished = false;
        private List<List<Tuple<double, double>>> _graph = new List<List<Tuple<double, double>>>();

        private ActivationFunctions.Del _input = ActivationFunctions.Input;
        private ActivationFunctions.Del _context = ActivationFunctions.Context;
        protected ActivationFunctions.Del1 _output = ActivationFunctions.SoftMax;
        private ActivationFunctions.Del _outGradient = GradientFunctions.Logistic;

        public List<List<double>> Inputs = new List<List<double>>();
        public List<List<double>> Correct = new List<List<double>>();
        public int Classes { get; set; }
        public int Epochs { get; set; }
        public bool Debug { get { return _debug; } set { _debug = value; } }
        public bool Finished { get { return _finished; } set { _finished = value; } }
        public ActivationFunctions.Del Input { get { return _input; } set { value = _input; } }
        public ActivationFunctions.Del Context { get { return _context; } set { value = _context; } }
        public ActivationFunctions.Del1 Output { get { return _output; } set { value = _output; } }
        public ActivationFunctions.Del OutGradient { get { return _outGradient; } set { value = _outGradient; } }

        public List<double> Errors = new List<double>();

        public List<List<Tuple<double, double>>> Graph { get { return _graph; } set { _graph = value; } }
        public List<int> IgnoreCol { get; set; }

        public Network() { }
        public Network(int layerCount, int neuronCount)
        {
            for(int l=0;l<layerCount;l++)
            {
                Layer layer = new Layer();
                for(int n=0;n<neuronCount;n++)
                {
                    Neuron neuron = new Neuron();
                    layer.Add(neuron);
                }
                this.Add(layer);
            }
        }

        public bool Run() { return false; }
        public bool Train() { return false; }

        public void GetGraph() { }

        public void InitializeStd(int[] nodes, Delegate activation, Delegate gradient, double min = -1, double max = 1)
        {

            foreach (var layer in this.Where(l => l.Type == LayerType.Hidden || l.Type == LayerType.Input))
            {
                foreach (var next in this.Where(l => ((l.Type == LayerType.Hidden) || (l.Type == LayerType.Output)) && (l.Index == layer.Index + 1)))
                {
                    layer.NextLayers.Add(next);
                    next.PreviousLayers.Add(layer);
                }

                foreach (var ctxlayer in this.Where(l => l.Type == LayerType.Context && l.Index == layer.Index))
                {
                    layer.NextLayers.Add(ctxlayer);
                    layer.PreviousLayers.Add(ctxlayer);
                    ctxlayer.NextLayers.Add(layer);
                    ctxlayer.PreviousLayers.Add(layer);
                }
            }


            

            Parallel.ForEach(this.Where(x => x.Type == LayerType.Input),
                layer =>
                {
                    InitializeLayer(layer.Index, Inputs[0].Count, Input, Input);
                });

            Parallel.ForEach(this.Where(x => x.Type == LayerType.Hidden),
                layer =>
                {
                    InitializeLayer(layer.Index, nodes[layer.Index-1], activation, gradient);
                });

            Parallel.ForEach(this.Where(x => x.Type == LayerType.Context),
                layer =>
                {
                    InitializeLayer(layer.Index+1, nodes[layer.Index-1], Context, Context);
                });

            Parallel.ForEach(this.Where(x => x.Type == LayerType.Output),
                layer =>
                {
                    InitializeLayer(layer.Index, Classes, Output, OutGradient);
                });

            Parallel.ForEach(this,
                layer =>
                {
                    layer.StandardConnectNeurons(min, max);
                });
        }

        public void InitializeLayer(int index, int nodes, Delegate activation, Delegate gradient)
        {
            Neuron[] layerNeuron = new Neuron[nodes];
            Parallel.For(0, nodes, i =>
            {
                try
                {
                    layerNeuron[i] = new Neuron(activation, gradient)
                    {
                        Name = string.Format("L{0}_N{1}", index, i),
                    };
                }
                catch
                {
                    Console.WriteLine("Cannot create Neuron at index: {0}", i);
                }
            });
            this[index].AddRange(layerNeuron.ToList());
        }

        private string[] RemoveIndex(string[] input)
        {
            var ret = input.ToList();
            foreach (int index in IgnoreCol)
            {
                try
                {
                    ret.RemoveAt(index);
                }
                catch
                {
                    continue;
                }
            }
            return ret.ToArray();
        }

        public void SetCorrect(string filename, int classes=1)
        {
            Correct = new List<List<double>>();
            using (var sr = new StreamReader(filename))
            {
                while (!sr.EndOfStream)
                {
                    List<double> rv = new List<double>();
                    string pattern = "[a-zA-z]+";
                    string input = sr.ReadLine();
                    string[] istr = Regex.Split(input, pattern);
                    istr = RemoveIndex(istr);
                    if (istr.Length > 1 || (input.Length <= 1))
                    {
                        for (int i = 0; i < classes; i++)
                            rv.AddRange(istr.Select(x => Double.Parse(x)).ToArray());
                    }
                    else
                    {
                        for (int i = 0; i < classes; i++)
                        {
                            byte[] inb = Encoding.ASCII.GetBytes(input);
                            rv.AddRange(inb.Select(x => (double)x));
                        }
                    }
                    Correct.Add(rv);
                }
            }
        }

        public void SetInputs(string filename, int cluster, bool normalize=true)
        {
            Inputs = new List<List<double>>();
            using (var sr = new StreamReader(filename))
            {
                while (!sr.EndOfStream)
                {
                    List<double> rv = new List<double>();
                    string pattern = "[ |,a-zA-z]+";
                    for (int i = 0; i < cluster; i++)
                    {
                        string input = sr.ReadLine();
                        string[] istr = Regex.Split(input, pattern);
                        istr = RemoveIndex(istr);
                        if (istr.Length > 1)
                        {
                            rv.AddRange(istr.Select(x => Double.Parse(x)).ToArray());

                        }
                        else
                        {
                            byte[] inb = Encoding.ASCII.GetBytes(input);
                            rv.AddRange(inb.Select(x => (double)x));
                        }
                    }
                    Inputs.Add(rv);
                }
            }

            if (normalize)
            {
                List<double> distinct = Inputs.SelectMany(x => x.Select(y => y).Distinct()).Distinct().ToList();
                Parallel.ForEach(Inputs, input =>
                {
                    for(int i=0; i<input.Count; i++)
                    {
                        input[i] = distinct.IndexOf(input[i]) + 0.1; //offset so zeroes don't ruin data
                    }
                });
            }
        }

        public void UpdateAllWeights(int batchCount=1)
        {
            Parallel.ForEach(this.SelectMany(x => x.SelectMany(y => y.InterOut)).ToList(),
                inter =>
                {
                    inter.Weight += (inter.Update / batchCount);
                    inter.Update = 0.0d;
                }
            );
        }


        public int _correct = 0;
        public int _classified = 0;
        public int _overPredict = 0;
        public int _underPredict = 0;
        public double GetOutputAccuracy()
        {
            var neuronErrors = this.Where(x => x.Type == LayerType.Output).SelectMany(x => x.Select(y => y.Error)).ToArray();
            double sumError = ActivationFunctions.SumSquaredError(neuronErrors);            
            Errors.Add(sumError);

            double max = this.Where(x => x.Type == LayerType.Output).SelectMany(x => x.Select(y => y.Prediction)).Max();

            Parallel.ForEach(
                this.Where(x => x.Type == LayerType.Output).SelectMany(x => x.Select(y => y)),
                x =>
                {
                    if (x.Prediction >= max)
                    {
                        if (x.Expected > _barrierTrain)
                        {
                            Interlocked.Increment(ref _correct);
                            Interlocked.Increment(ref _classified);
                        }
                        else
                            Interlocked.Increment(ref _overPredict);
                    }
                    else
                    {
                        if (x.Expected < _barrierTrain)
                            Interlocked.Increment(ref _correct);
                        else
                            Interlocked.Increment(ref _underPredict);
                    }
                }
            );
            return ((double)_correct / (_correct + _overPredict + _underPredict)) * 100.0d;
        }
    }
}
