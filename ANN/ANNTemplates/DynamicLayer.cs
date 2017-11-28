using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace ANN.ANNTemplates
{
    public class DynamicLayer : List<DynamicNeuron>
    {
        public int Dimensions { get; set; }
        public string Label { get; set; }
        public int Index { get; set; }
        public LayerType Type { get; set; }
        public List<DynamicLayer> PreviousLayers = new List<DynamicLayer>();
        public List<DynamicLayer> NextLayers = new List<DynamicLayer>();

        private List<List<double>> _inputs = new List<List<double>>();

        static Random rnd = new Random();


        public void SOMForward(int radius = 1, bool randomstep=false)
        {
            try
            {
                if (Type == LayerType.Input)
                {
                    Parallel.ForEach(this, neuron =>
                    {
                        neuron.OutInter = new List<DynamicInterconnect>()
                        {
                            new DynamicInterconnect() {ForwardPass = neuron.Internal.Weights}
                        };
                    });

                    _inputs = this.Select(n => n.Internal.Weights).ToList();
                    foreach (var layer in NextLayers)
                    {
                        layer._inputs = _inputs;
                    }
                }

                else if (Type == LayerType.Output)
                {
                    //int r = rnd.Next(Count);
                    //var neuron = this[r];
                    int idist = 0;
                    int ineur = 0;
                    if (randomstep)
                    {
                        int r = rnd.Next(Count);
                        var neuron = this[r];
                        List<double> distance = new List<double>();
                        foreach (var point1 in _inputs)
                        {
                            var point2 = neuron.Internal.Weights;
                            double dist = Math.Sqrt(point1.Zip(point2, (a, b) => (a - b) * (a - b)).Sum());
                            distance.Add(dist);
                        }

                        idist = distance.IndexOf(distance.Min());
                    }
                    else
                    {
                        List<List<double>> distances = new List<List<double>>();
                        bool found = false;
                        foreach (var neuron in this)
                        {
                            List<double> distance = new List<double>();
                            foreach (var point1 in _inputs)
                            {
                                var point2 = neuron.Internal.Weights;
                                double dist = Math.Sqrt(point1.Zip(point2, (a, b) => (a - b) * (a - b)).Sum());
                                distance.Add(dist);
                            }
                            distances.Add(distance);
                            
                        }

                        double min = distances[0].Sum();
                        for (int i = 1; i < Count; i++)
                        {
                            double tsum = distances[i].Sum();
                            min = (tsum > min) ? tsum : min;
                            idist = (tsum == min) ? i : idist;
                        }
                        idist = distances[idist].IndexOf(distances[idist].Min());

                    }

                    for (int i = -1 * radius; i <= radius; i++)
                    {
                        int newi = idist + i;
                        //int newi = ineur + i;
                        if (newi < 0)
                        {
                            newi = Count - Math.Abs(newi);
                        }
                        else if (newi >= Count)
                        {
                            newi = newi - Count; 
                        }

                        double limit = (newi == idist) ? 1: 0.005;

                        List<double> updates = new List<double>();
                        for (int j = 0; j < _inputs[idist].Count; j++)
                            updates.Add(_inputs[idist][j] - this[idist].Weights[j]);//*/
                        /*for (int j = 0; j < _inputs[idist].Count; j++)
                            updates.Add(_inputs[idist][j] - this[newi].Weights[j]);*/

                        updates.ForEach(x => x = x * limit);

                        this[newi].Update = updates.ToArray();

                        if (Math.Abs(this[newi].Update.Sum()) >= 0.01)
                            this[newi].Ignore = false;
                    }

                }

                else
                {

                }
            }
            catch (Exception e)
            {
                Debug.WriteLine("Exception: {0}", e.ToString());
            }
        }

        public void SOMForward(bool runthis, int radius =1, bool randomstep = false)
        {
            try
            {
                if (Type == LayerType.Input)
                {
                    Parallel.ForEach(this, neuron =>
                    {
                        neuron.OutInter = new List<DynamicInterconnect>()
                        {
                            new DynamicInterconnect() {ForwardPass = neuron.Internal.Weights}
                        };
                    });

                    _inputs = this.Select(n => n.Internal.Weights).ToList();
                    foreach (var layer in NextLayers)
                    {
                        layer._inputs = _inputs;
                    }
                }

                else if (Type == LayerType.Output)
                {
                    //int r = rnd.Next(Count);
                    //var neuron = this[r];
                    int idist = 0;
                    int ineur = 0;
                    List<int> nindex = new List<int>();

                    if (randomstep)
                    {
                        int r = rnd.Next(Count);
                        var neuron = this[r];
                        List<double> distance = new List<double>();
                        foreach (var point1 in _inputs)
                        {
                            var point2 = neuron.Internal.Weights;
                            double dist = Math.Sqrt(point1.Zip(point2, (a, b) => (a - b) * (a - b)).Sum());
                            distance.Add(dist);
                        }

                        idist = distance.IndexOf(distance.Min());
                        nindex.Add(idist);
                    }
                    else
                    {
                        List<List<double>> distances = new List<List<double>>();
                        bool found = false;
                        foreach (var neuron in this)
                        {
                            List<double> distance = new List<double>();
                            foreach (var point1 in _inputs)
                            {
                                var point2 = neuron.Internal.Weights;
                                double dist = Math.Sqrt(point1.Zip(point2, (a, b) => (a - b) * (a - b)).Sum());
                                distance.Add(dist);
                            }
                            distances.Add(distance);
                            nindex.Add(distance.IndexOf(distance.Min()));
                        }

                        if (!found)
                            return;

                        double min = distances[0].Sum();
                        for (int i = 1; i < Count; i++)
                        {
                            double tsum = distances[i].Sum();
                            min = (tsum > min) ? tsum : min;
                            idist = (tsum == min) ? i : idist;
                        }
                        idist = distances[idist].IndexOf(distances[idist].Min());

                    }

                    for (int ind = 0; ind < nindex.Count; ind++)
                    {
                        ineur = ind;
                        idist = nindex[ind];
                        for (int i = -1 * radius; i <= radius; i++)
                        {
                            //int newi = idist + i;
                            int newi = ineur + i;
                            if (newi < 0)
                            {
                                newi = Count - Math.Abs(newi);
                            }
                            else if (newi >= Count)
                            {
                                newi = newi - Count;
                            }

                            double limit = (newi == idist) ? 1 : 0.005;

                            List<double> updates = new List<double>();
                            /*for (int j = 0; j < _inputs[idist].Count; j++)
                                updates.Add(_inputs[idist][j] - this[idist].Weights[j]);//*/
                            for (int j = 0; j < _inputs[idist].Count; j++)
                                updates.Add(_inputs[idist][j] - this[newi].Weights[j]);

                            updates.ForEach(x => x = x * limit);

                            this[newi].Update = updates.ToArray();

                            if (Math.Abs(this[newi].Update.Sum()) >= 0.01)
                                this[newi].Ignore = false;
                        }
                    }

                }

                else
                {

                }
            }
            catch (Exception e)
            {
                Debug.WriteLine("Exception: {0}", e.ToString());
            }
        }

        public DynamicLayer(int neurons, int dimensions=1)
        {
            Dimensions = dimensions;
            if (dimensions == 1)
            {
                for(int i = 0; i < neurons; i++)
                {
                    this.Add(new DynamicNeuron());
                }
            }

            else
            {
                Console.WriteLine("MultiDimensionality Not implemented");
            }

        }
    }

}
