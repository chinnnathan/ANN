using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ANN.ANNTemplates;
using ANN.Utils;
using System.Diagnostics;

namespace ANN.Networks
{
    public class Backpropagation : Network
    {
        private bool _debug = true;

        public Backpropagation() { }

        public Backpropagation(int outputnodes, int hiddenlayers=1)
        {
            Layer input = new Layer()
            {
                Index = 0,
                Label = "Input",
                Type = LayerType.Input,
            };
            Layer output = new Layer()
            {
                Index = hiddenlayers + 1,
                Label = "Output",
                Type = LayerType.Output,
            };
            Add(input);

            for(int i=0; i<hiddenlayers; i++)
            {
                Add(new Layer()
                {   Index = i + 1,
                    Label = string.Format("HiddenLayer_{0}", i),
                    Type = LayerType.Hidden,
                });
            }
            Add(output);
        }

        new public bool Run()
        {
            int index = 0;
            foreach (var input in Inputs)
            {
                Parallel.For(0, this[this.Count - 1].Count, i =>
                {
                    this[this.Count - 1][i].Expected = (Correct[index][i] == (i + 1)) ? 1.0 : -1.0;
                });

                Parallel.For(0, this[0].Count, i =>
                {
                    this[0][i].Prediction = input[i];
                    Parallel.For(0, this[0][i].InterOut.Count, j =>
                    {
                        this[0][i].InterOut[j].IValue = Inputs[index][0];
                    });
                });

                foreach (var layer in this.Where(x => x.Index != this.Count))
                {
                    layer.FeedForward();
                }
                index++;
            }

            return true;
        }

        private bool RunBackprop(int index)
        {
            
            Parallel.For(0, this[this.Count - 1].Count, i =>
            {
                this[this.Count - 1][i].Expected = (Correct[index][i] == (i + 1)) ? 1.0 : -1.0;
            });

            Parallel.ForEach(this.Where(x=>x.Type == LayerType.Input), layer =>
            {
                Parallel.For(0, layer.Count, i =>
                {
                    layer[i].InterIn[0].IValue = Inputs[index][i];
                    layer[i].InterIn[0].DValue = Inputs[index][i];
                    layer[i].InterIn[0].FValue = Inputs[index][i];
                    layer[i].Prediction = Inputs[index][i];
                });
            });

            foreach (var layer in this.Where(x => x.Index != this.Count))
            {
                layer.FeedForward();
            }

            foreach (var layer in this.Select(x=>x).Reverse())
            {
                layer.BackPropagate();
            }
            
            return true;
        }

        new public bool Train()
        {
            if (!_debug) // want to minimize the amount of operations per iteration
            {
                for (int epoch = 0; epoch < Epochs; epoch++)
                {
                    Console.Write("\r[{0}] Starting Epoch {1}", DateTime.Now, epoch + 1);
                    for (int datasetlabel = 0; datasetlabel < Correct.Count; datasetlabel++)
                    {
                        RunBackprop(datasetlabel);
                    }
                }
            }
            else
            {
                using (var sw = new StreamWriter("outputs_Backpropagation_.csv".AppendTimeStamp()))
                {
                    sw.Write("Epoch,");
                    sw.WriteLine(this[0].PrintCSVHeader());
                    for (int epoch = 0; epoch < Epochs; epoch++)
                    {
                        for (int datasetlabel = 0; datasetlabel < Correct.Count; datasetlabel++)
                        {
                            RunBackprop(datasetlabel);
                            GetOutputAccuracy();
                            foreach (var layer in this)
                            {
                                foreach (var neuron in this)
                                {
                                    sw.Write("{0},", epoch);
                                    sw.WriteLine(neuron.PrintCSVLine());
                                }
                            }
                            /*
                                foreach (var neuron in layer)
                                    sw.WriteLine(neuron.PrintCSVLine());*/
                        }
                        Console.WriteLine("\r[{0}] Epoch {1} - Accuracy: {2} Correct: {3} Over: {4} Under:{5}", DateTime.Now, epoch + 1,
                            GetOutputAccuracy(), _correct, _overPredict, _underPredict);
                    }
                }
            }
            Console.WriteLine("\nTraining Finished");
            return true;
        }

    }
}
