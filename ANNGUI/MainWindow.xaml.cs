using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Wpf.CartesianChart.MaterialCards;
using ANN;
using ANN.Utils;
using Wpf.CartesianChart.ScatterPlot;

namespace ANNGUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private object _lock = new object();

        ANN.ANN ann = new ANN.ANN();
        private List<string> args = new List<string>();

        internal static MainWindow main;
        internal List<List<Tuple<double,double>>> GetWeights { get { return ann.Graph; } }
        internal int GetClasses { get { return ann.Classes; } }

        public MainWindow()
        {
            main = this;
            InitializeComponent();

            foreach (var arg in ann.GetVariableNames())
            {
                LVData lvd = new LVData(arg.Item1, arg.Item2);
                variableSet.Items.Add(lvd);
            }
        }

        private void TrainNetworkBtn_Click(object sender, RoutedEventArgs e)
        {
            ann.RunProgram(args.ToArray());
        }

        private void OutputAddLine(string text)
        {
            OutputText.AppendText(text);
            OutputText.AppendText("\u2028"); //Linebreak, not paragraph
            OutputText.ScrollToEnd();
        }

        private void variableSet_CellEditEnding(object sender, DataGridCellEditEndingEventArgs e)
        {
            var cd = ((DataGrid)sender).CurrentCell;
        }

        private void TextBox_LostFocus(object sender, RoutedEventArgs e)
        {
            var cd = ((TextBox)sender);
        }

        private bool _bail = false;
        private bool _running = false;

        private async Task UpdateMap()
        {
            do
            {
                ScatterExample.scatter.Series = ann.Graph;
                ScatterExample.scatter.GraphNewData();
                await Task.Delay(50);
            } while (!ann.Finished);
        }

        private async Task RunNetwork()
        {
            OutputAddLine("Running Network Now");
            await Task.Run(() => ann.RunProgram(args.ToArray()));
        }

        private async void GraphNetworkBtn_Click(object sender, RoutedEventArgs e)
        {
            //Run 1000? epochs, graph 100 at a time
            tabControl.SelectedIndex = 2;
            int radius = 5;
            int epochs = 0;
            int epi = 500;
            double min = 10000;
            double max = 30000;
            ann = new ANN.ANN();
            args.AddRange(new List<string>() { "Epochs", epi.ToString(), "NewNetwork", "false", "Min", min.ToString(), "Max", max.ToString(), "Radius", radius.ToString() });
            OutputAddLine(string.Format("[{0}] Start", DateTime.Now));
            /*do
            {
                _running = true;
                await Task.WhenAll(
                    Task.Run(()=>ann.RunProgram(args.ToArray())),
                    Task.Run(() => _running = false)
                    );
                //await Task.Run(() => ann.RunProgram(args.ToArray()));

                ScatterExample.scatter.Series = GetWeights;
                ScatterExample.scatter.GraphNewData();

                if (radius > 0)
                    radius--;
                args.RemoveAt(args.Count - 1);
                args.Add(radius.ToString());
                epochs += epi;
            } while (!ann.Finished && !_bail);*/

            //await Task.Run(() => ann.RunProgram(args.ToArray()));

            var tasks = new List<Task>();
            tasks.Add(RunNetwork());
            tasks.Add(UpdateMap());
            await Task.WhenAll(tasks);

            if (_bail)
            {
                OutputAddLine(string.Format("BAILED", DateTime.Now, epochs));
                OutputAddLine(string.Format("[{0}] Epochs: {1}", DateTime.Now, epochs));
            }
            else
                OutputAddLine(string.Format("[{0}] Epochs: {1}", DateTime.Now, epochs));
        }

        private void Reset_Click(object sender, RoutedEventArgs e)
        {
            _bail = true;
            while (_running)
            {
                Task.Delay(10);
            }
            _bail = false;
            ann = new ANN.ANN();
        }
    }

    public struct LVData
    {
        private string _name;
        private dynamic _value;

        public string Name { get { return _name; } set { _name = value; } }
        public dynamic Value { get { return _value; } set { _name = value; } }

        public LVData(string varName, dynamic value)
        {
            _name = varName;
            _value = value;
        }
    }

}
