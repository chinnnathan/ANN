using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using ANN.Utils;

namespace ANN.ANNTemplates
{
    public class Network : List<Layer>
    {
        public List<List<double>> Inputs = new List<List<double>>();
        public List<List<double>> Correct = new List<List<double>>();
        public int Classes { get; set; }
        public int Epochs { get; set; }

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

        public void InitializeStd(int nodes, Delegate activation, Delegate gradient, double min=-1, double max=1)
        {
            for (int i = 1; i < this.Count - 1; i++)
            {
                this[i].NextLayer = this[i + 1];
                this[i].PreviousLayer = this[i - 1];
            }

            this[0].NextLayer = this[1];
            this[this.Count - 1].PreviousLayer = this[this.Count - 2];

            Parallel.For(1, this.Count - 1, i =>
            {
                InitializeLayer(i, nodes, activation, gradient);
            });//Last Async call appropriate, after this order matters

            ActivationFunctions.Del input = ActivationFunctions.Input;
            ActivationFunctions.Del1 softmax = ActivationFunctions.SoftMax;
            ActivationFunctions.Del2 ssoftmax = GradientFunctions.SoftMax;
            InitializeLayer(0, Inputs[0].Count, input, input); //no real activation function
            InitializeLayer(this.Count - 1, Classes, softmax, ssoftmax);

            foreach (var layer in this)
            {
                layer.StandardConnectNeurons(min, max);
            }
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
                    Debug.WriteLine("Cannot create Neuron at index: {0}", i);
                }
            });
            this[index].AddRange(layerNeuron.ToList());
        }


        public void SetCorrect(string filename, int classes=1)
        {
            using (var sr = new StreamReader(filename))
            {
                while (!sr.EndOfStream)
                {
                    List<double> rv = new List<double>();
                    string pattern = "[a-zA-z]+";
                    string input = sr.ReadLine();
                    string[] istr = Regex.Split(input, pattern);
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

        public void SetInputs(string filename, int cluster=1)
        {
            using (var sr = new StreamReader(filename))
            {
                while (!sr.EndOfStream)
                {
                    List<double> rv = new List<double>();
                    string pattern = "[a-zA-z]+";
                    string input = sr.ReadLine();
                    string[] istr = Regex.Split(input, pattern);
                    if (istr.Length > 1)
                    {
                        for (int i = 0; i < cluster; i++)
                            rv.AddRange(istr.Select(x => Double.Parse(x)).ToArray());

                    }
                    else
                    {
                        for (int i = 0; i < cluster; i++)
                        {
                            byte[] inb = Encoding.ASCII.GetBytes(input);
                            rv.AddRange(inb.Select(x => (double)x));
                        }
                    }
                    Inputs.Add(rv);
                }
            }
        }

        public double _correct = 0;
        public double _overPredict = 0;
        public double _underPredict = 0;
        public double GetOutputAccuracy()
        {
            foreach (var outlayer in this.Where(x=>x.Type == LayerType.Output))
            {
                double max = outlayer.Select(x => x.Prediction).Max();
                foreach (var neuron in outlayer)
                {
                    if ((neuron.Expected > 0) && (neuron.Prediction < max))
                        _underPredict++;
                    else if ((neuron.Expected < 0) && (neuron.Prediction == max))
                        _overPredict++;
                    else
                        _correct++;
                }
            }

            return (_correct / (_correct + _overPredict + _underPredict));
        }
    }
}
