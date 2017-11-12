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
    public class Recurrent : Network
    {
        public Recurrent() { }

        public Recurrent(int outputnodes, int hiddenlayers = 1)
        {
            Layer input = new Layer()
            {
                Index = 0,
                Label = "Input",
                Type = LayerType.Input,
            };
            Add(input);

            Layer output = new Layer()
            {
                Index = hiddenlayers + 1,
                Label = "Output",
                Type = LayerType.Output,
            };
            Add(output);

            for (int i = 0; i < hiddenlayers; i++)
            {
                Add(new Layer()
                {
                    Index = i + 1,
                    Label = string.Format("HiddenLayer_{0}", i),
                    Type = LayerType.Hidden,
                });

                Add(new Layer()
                {
                    Index = i + 1,
                    Label = string.Format("ContextLayer_{0}", i),
                    Type = LayerType.Context,
                });
            }
        }

        new public bool Run()
        {
            Console.WriteLine("[{0}] Run  -> Testing Data Sets: {1} Classes:{2}", DateTime.Now, Correct.Count, Classes);
            double acc = 0;
            for (int index = 0; index < Correct.Count; index++)
            {
                Parallel.For(0, this[this.Count - 1].Count, i =>
                {
                    this[this.Count - 1][i].Expected = (Correct[index][i] == (i + 1)) ? _correctTrain : _incorrectTrain;
                });

                Parallel.ForEach(this.Where(x => x.Type == LayerType.Input), layer =>
                {
                    Parallel.For(0, layer.Count, i =>
                    {
                        layer[i].InterIn[0].IValue = Inputs[index][i];
                        layer[i].InterIn[0].DValue = Inputs[index][i];
                        layer[i].InterIn[0].FValue = Inputs[index][i];
                        layer[i].Input = Inputs[index][i];
                    });
                });

                foreach (var layer in this.Where(x => x.Index != this.Count))
                {
                    layer.FeedForward();
                    acc = GetOutputAccuracy();
                }
                Console.Write("\r[{0}] - Accuracy: {1:N4}% Correct: {2} Over: {3} Under:{4} Classified:{5}", DateTime.Now, acc, _correct, _overPredict, _underPredict, _classified);
            }
            Console.WriteLine("\n[{0}] Finish Run -> Testing Data Sets: {1} Classes:{2}", DateTime.Now, Correct.Count, Classes);

            return true;
        }

        private bool RunRecurrent(int index)
        {

            Parallel.For(0, this[this.Count - 1].Count, i =>
            {
                this[this.Count - 1][i].Expected = (Correct[index][i] == (i + 1)) ? _correctTrain : _incorrectTrain;
            });

            Parallel.ForEach(this.Where(x => x.Type == LayerType.Input), layer =>
            {
                Parallel.For(0, layer.Count, i =>
                {
                    layer[i].InterIn[0].IValue = Inputs[index][i];
                    layer[i].InterIn[0].DValue = Inputs[index][i];
                    layer[i].InterIn[0].FValue = Inputs[index][i];
                    layer[i].Input = Inputs[index][i];
                });
            });

            foreach (var layer in this.Where(x => x.Index != this.Count))
            {
                layer.FeedForward();
            }

            foreach (var layer in this.Select(x => x).Reverse())
            {
                layer.BackPropagate();
            }

            return true;
        }

        public bool Train(bool debug = true)
        {
            Console.WriteLine("[{0}] Starting Training -> Epochs: {1} Training Data Sets: {2} Classes:{3}", DateTime.Now, Epochs, Correct.Count, Classes);
            double acc = 0;
            if (!debug) // want to minimize the amount of operations per iteration
            {
                for (int epoch = 0; epoch < Epochs; epoch++)
                {
                    for (int datasetlabel = 0; datasetlabel < Correct.Count; datasetlabel++)
                    {
                        RunRecurrent(datasetlabel);
                        UpdateAllWeights();
                    }
                    Console.Write("\r[{0}] Epoch {1} - Accuracy: {2:N4}% Correct: {3} Over: {4} Under:{5} Classified:{6}", DateTime.Now, epoch + 1,
                                GetOutputAccuracy(), _correct, _overPredict, _underPredict, _classified);
                }
            }
            else
            {
                using (var sw = new StreamWriter("outputs_Recurrent_.csv".AppendTimeStamp()))
                {
                    sw.WriteLine(",Errors_{0},Accuracy,Correct,Classification,OverPrediction,", string.Join(",Errors_", Enumerable.Range(1, Correct.Count()).ToArray()));
                    for (int epoch = 0; epoch < Epochs; epoch++)
                    {
                        sw.Write("{0},", epoch);
                        for (int datasetlabel = 0; datasetlabel < Correct.Count; datasetlabel++)
                        {
                            RunRecurrent(datasetlabel);
                            acc = GetOutputAccuracy();
                            UpdateAllWeights();
                            sw.Write("{0},", Errors.Last());

                        }
                        sw.WriteLine("{0},{1},{2},{3},", acc, _correct, _classified, _overPredict);
                        Console.Write("\r[{0}] Epoch {1} - Accuracy: {2:N4}% Correct: {3} Over: {4} Under:{5} Classified:{6}", DateTime.Now, epoch + 1,
                            acc, _correct, _overPredict, _underPredict, _classified);
                    }
                }
            }
            Console.WriteLine("\nTraining Finished");
            return true;
        }

    }
}