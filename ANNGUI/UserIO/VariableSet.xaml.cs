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

namespace ANNGUI.UserIO
{
    /// <summary>
    /// Interaction logic for VariableSet.xaml
    /// </summary>
    public partial class VariableSet : UserControl
    {
        private List<string> _variables = new List<string>() { "Temporary", "Debugging" };

        public VariableSet()
        {
            InitializeComponent();
            int i = 0;
            foreach (var variable in _variables)
            {
                Model item = new Model(variable, -1, "TextBox");
                listView.Items.Add(variable);
                listView.Items.Add(i);
                i++;
            }
        }
    }

    public class Model
    {
        private dynamic _value;
        public string Name { get; set; }
        public string Value { get { return string.Format("{0}", _value); } }
        public string View { get; set; }

        public Model() { }
        public Model(string name, dynamic value, string view)
        {
            _value = value;
            Name = name;
            View = view;
        }

    }

    public class DynamicDataTemplateSelector : DataTemplateSelector
    {
        public override DataTemplate
            SelectTemplate(object item, DependencyObject container)
        {
            FrameworkElement element = container as FrameworkElement;

            if (element != null && item != null && item is Task)
            {
                Model model = item as Model;

                return (DataTemplate)element.FindResource(model.View + "Template");
            }

            return null;
        }
    }
}
