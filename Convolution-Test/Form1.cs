using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ANN_Lib;

namespace Convolution_Test
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        float[,] filter01 = new float[3, 3];
        float[,] filter02 = new float[3, 3];
        float[,] filter03 = new float[3, 3];
        float[,] filter04 = new float[3, 3];
        float[,] filter05 = new float[3, 3];

        float[,] arr01;
        float[,] arr02;

        ANN_Lib.CNN cnn = new CNN();

        private void button1_Click(object sender, EventArgs e)
        {
            uc1.ReadImage();
            uc1.Data = cnn.AddPadding(uc1.Data, 3);
            uc1.init(18, 18, 15, 15);

            filter01[0, 0] = 1; filter01[1, 0] = -1; filter01[2, 0] = -1;
            filter01[0, 1] = -1; filter01[1, 1] = 1; filter01[2, 1] = -1;
            filter01[0, 2] = -1; filter01[1, 2] = -1; filter01[2, 2] = 1;
            uc2.Data = filter01; uc2.init(3, 3, 8, 8);

            filter02[0, 0] = -1; filter02[1, 0] = -1; filter02[2, 0] = 1;
            filter02[0, 1] = -1; filter02[1, 1] = 1; filter02[2, 1] = -1;
            filter02[0, 2] = 1; filter02[1, 2] = -1; filter02[2, 2] = -1;
            uc3.Data = filter02; uc3.init(3, 3, 8, 8);

            filter03[0, 0] = -1; filter03[1, 0] = -1; filter03[2, 0] = -1;
            filter03[0, 1] = 1; filter03[1, 1] = 1; filter03[2, 1] = 1;
            filter03[0, 2] = -1; filter03[1, 2] = -1; filter03[2, 2] = -1;
            uc4.Data = filter03; uc4.init(3, 3, 8, 8);

            filter04[0, 0] = -1; filter04[1, 0] = 1; filter04[2, 0] = -1;
            filter04[0, 1] = -1; filter04[1, 1] = 1; filter04[2, 1] = -1;
            filter04[0, 2] = -1; filter04[1, 2] = 1; filter04[2, 2] = -1;
            uc5.Data = filter04; uc5.init(3, 3, 8, 8);

            filter05[0, 0] = 1; filter05[1, 0] = -1; filter05[2, 0] = 1;
            filter05[0, 1] = -1; filter05[1, 1] = 1; filter05[2, 1] = -1;
            filter05[0, 2] = 1; filter05[1, 2] = -1; filter05[2, 2] = 1;
            uc6.Data = filter05; uc6.init(3, 3, 8, 8);
        }
        

       
        private void button3_Click(object sender, EventArgs e)
        {
            uc29.Data = cnn.RemovePadding(uc29.Data, 3);
            uc29.SaveImage();
        }

        private void button4_Click(object sender, EventArgs e)
        {
            float[,] data01 = cnn.Conv(uc1.Data, filter01);
            //uc6.Normalize(data01, 1.0f, -1.0f);
            uc7.Data = cnn.ReLu(data01);
            uc7.init(16, 16, 5, 5);

            float[,] data02 = cnn.Conv(uc1.Data, filter02);
            //uc7.Normalize(data02, 1.0f, -1.0f);
            uc8.Data = cnn.ReLu(data02);
            uc8.init(16, 16, 5, 5);

            float[,] data03 = cnn.Conv(uc1.Data, filter03);
            //uc8.Normalize(data03, 1.0f, -1.0f);
            uc9.Data = cnn.ReLu(data03);
            uc9.init(16, 16, 5, 5);

            float[,] data04 = cnn.Conv(uc1.Data, filter04);
            //uc9.Normalize(data04, 1.0f, -1.0f);
            uc10.Data = cnn.ReLu(data04);
            uc10.init(16, 16, 5, 5);

            float[,] data05 = cnn.Conv(uc1.Data, filter05);
            //uc9.Normalize(data04, 1.0f, -1.0f);
            uc11.Data = cnn.ReLu(data05);
            uc11.init(16, 16, 5, 5);



            uc12.Data = cnn.Pooling(uc7.Data, 2, 2);
            uc12.init(8, 8, 7, 7);

            uc13.Data = cnn.Pooling(uc8.Data, 2, 2);
            uc13.init(8, 8, 7, 7);

            uc14.Data = cnn.Pooling(uc9.Data, 2, 2);
            uc14.init(8, 8, 7, 7);

            uc15.Data = cnn.Pooling(uc10.Data, 2, 2);
            uc15.init(8, 8, 7, 7);

            uc16.Data = cnn.Pooling(uc11.Data, 2, 2);
            uc16.init(8, 8, 7, 7);



            uc17.Data = cnn.Pooling(uc12.Data, 2, 2);
            uc17.init(4, 4, 14, 14);

            uc18.Data = cnn.Pooling(uc13.Data, 2, 2);
            uc18.init(4, 4, 14, 14);

            uc19.Data = cnn.Pooling(uc14.Data, 2, 2);
            uc19.init(4, 4, 14, 14);

            uc20.Data = cnn.Pooling(uc15.Data, 2, 2);
            uc20.init(4, 4, 14, 14);

            uc21.Data = cnn.Pooling(uc16.Data, 2, 2);
            uc21.init(4, 4, 14, 14);



            uc22.Data = cnn.Pooling(uc17.Data, 2, 2);
            uc22.init(2, 2, 28, 28);

            uc23.Data = cnn.Pooling(uc18.Data, 2, 2);
            uc23.init(2, 2, 28, 28);

            uc24.Data = cnn.Pooling(uc19.Data, 2, 2);
            uc24.init(2, 2, 28, 28);

            uc25.Data = cnn.Pooling(uc20.Data, 2, 2);
            uc25.init(2, 2, 28, 28);

            uc26.Data = cnn.Pooling(uc21.Data, 2, 2);
            uc26.init(2, 2, 28, 28);



            arr01 = new float[1, 20]; ;

            int k = 0;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    arr01[0, k] = uc22.Data[j, i];
                    arr01[0, k + 4] = uc23.Data[j, i];
                    arr01[0, k + 8] = uc24.Data[j, i];
                    arr01[0, k + 12] = uc25.Data[j, i];
                    arr01[0, k + 16] = uc26.Data[j, i];
                    k++;
                }
            }
            uc27.Data = arr01;
            uc27.init(1, 20, 22, 22);
        }

        private void button5_Click(object sender, EventArgs e)
        {
            float[,] data01 = cnn.Conv(uc29.Data, filter01);
            //uc6.Normalize(data01, 1.0f, -1.0f);
            uc7.Data = cnn.ReLu(data01);
            uc7.init(16, 16, 5, 5);

            float[,] data02 = cnn.Conv(uc29.Data, filter02);
            //uc7.Normalize(data02, 1.0f, -1.0f);
            uc8.Data = cnn.ReLu(data02);
            uc8.init(16, 16, 5, 5);

            float[,] data03 = cnn.Conv(uc29.Data, filter03);
            //uc8.Normalize(data03, 1.0f, -1.0f);
            uc9.Data = cnn.ReLu(data03);
            uc9.init(16, 16, 5, 5);

            float[,] data04 = cnn.Conv(uc29.Data, filter04);
            //uc9.Normalize(data04, 1.0f, -1.0f);
            uc10.Data = cnn.ReLu(data04);
            uc10.init(16, 16, 5, 5);

            float[,] data05 = cnn.Conv(uc29.Data, filter05);
            //uc9.Normalize(data04, 1.0f, -1.0f);
            uc11.Data = cnn.ReLu(data05);
            uc11.init(16, 16, 5, 5);



            uc12.Data = cnn.Pooling(uc7.Data, 2, 2);
            uc12.init(8, 8, 7, 7);

            uc13.Data = cnn.Pooling(uc8.Data, 2, 2);
            uc13.init(8, 8, 7, 7);

            uc14.Data = cnn.Pooling(uc9.Data, 2, 2);
            uc14.init(8, 8, 7, 7);

            uc15.Data = cnn.Pooling(uc10.Data, 2, 2);
            uc15.init(8, 8, 7, 7);

            uc16.Data = cnn.Pooling(uc11.Data, 2, 2);
            uc16.init(8, 8, 7, 7);



            uc17.Data = cnn.Pooling(uc12.Data, 2, 2);
            uc17.init(4, 4, 14, 14);

            uc18.Data = cnn.Pooling(uc13.Data, 2, 2);
            uc18.init(4, 4, 14, 14);

            uc19.Data = cnn.Pooling(uc14.Data, 2, 2);
            uc19.init(4, 4, 14, 14);

            uc20.Data = cnn.Pooling(uc15.Data, 2, 2);
            uc20.init(4, 4, 14, 14);

            uc21.Data = cnn.Pooling(uc16.Data, 2, 2);
            uc21.init(4, 4, 14, 14);



            uc22.Data = cnn.Pooling(uc17.Data, 2, 2);
            uc22.init(2, 2, 28, 28);

            uc23.Data = cnn.Pooling(uc18.Data, 2, 2);
            uc23.init(2, 2, 28, 28);

            uc24.Data = cnn.Pooling(uc19.Data, 2, 2);
            uc24.init(2, 2, 28, 28);

            uc25.Data = cnn.Pooling(uc20.Data, 2, 2);
            uc25.init(2, 2, 28, 28);

            uc26.Data = cnn.Pooling(uc21.Data, 2, 2);
            uc26.init(2, 2, 28, 28);



            arr02 = new float[1, 20]; ;

            int k = 0;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    arr02[0, k] = uc22.Data[j, i];
                    arr02[0, k + 4] = uc23.Data[j, i];
                    arr02[0, k + 8] = uc24.Data[j, i];
                    arr02[0, k + 12] = uc25.Data[j, i];
                    arr02[0, k + 16] = uc26.Data[j, i];
                    k++;
                }
            }
            uc28.Data = arr02;
            uc28.init(1, 20, 22, 22);
        }
        private void button6_Click(object sender, EventArgs e)
        {
            uc29.ReadImage();
            uc29.Data = cnn.AddPadding(uc29.Data, 3);
            uc29.init(18, 18, 15, 15);

            filter01[0, 0] = 1; filter01[1, 0] = -1; filter01[2, 0] = -1;
            filter01[0, 1] = -1; filter01[1, 1] = 1; filter01[2, 1] = -1;
            filter01[0, 2] = -1; filter01[1, 2] = -1; filter01[2, 2] = 1;
            uc2.Data = filter01; uc2.init(3, 3, 8, 8);

            filter02[0, 0] = -1; filter02[1, 0] = -1; filter02[2, 0] = 1;
            filter02[0, 1] = -1; filter02[1, 1] = 1; filter02[2, 1] = -1;
            filter02[0, 2] = 1; filter02[1, 2] = -1; filter02[2, 2] = -1;
            uc3.Data = filter02; uc3.init(3, 3, 8, 8);

            filter03[0, 0] = -1; filter03[1, 0] = -1; filter03[2, 0] = -1;
            filter03[0, 1] = 1; filter03[1, 1] = 1; filter03[2, 1] = 1;
            filter03[0, 2] = -1; filter03[1, 2] = -1; filter03[2, 2] = -1;
            uc4.Data = filter03; uc4.init(3, 3, 8, 8);

            filter04[0, 0] = -1; filter04[1, 0] = 1; filter04[2, 0] = -1;
            filter04[0, 1] = -1; filter04[1, 1] = 1; filter04[2, 1] = -1;
            filter04[0, 2] = -1; filter04[1, 2] = 1; filter04[2, 2] = -1;
            uc5.Data = filter04; uc5.init(3, 3, 8, 8);

            filter05[0, 0] = 1; filter05[1, 0] = -1; filter05[2, 0] = 1;
            filter05[0, 1] = -1; filter05[1, 1] = 1; filter05[2, 1] = -1;
            filter05[0, 2] = 1; filter05[1, 2] = -1; filter05[2, 2] = 1;
            uc6.Data = filter05; uc6.init(3, 3, 8, 8);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            float Perception = 0.0f; int c = 0;
            for (int i = 0; i < arr01.GetLength(1); i++)
            {
                if (arr01[0, i] >= 1.0f)
                {
                    Perception += arr02[0, i]; c++;
                }

            }
            label1.Text = "" + Perception / (float)c;

            Perception = 0.0f; c = 0;
            for (int i = 0; i < arr01.GetLength(1); i++)
            {
                if (arr02[0, i] >= 1.0f)
                {
                    Perception += arr01[0, i]; c++;
                }

            }
            label2.Text = "" + Perception / (float)c;
        }  

       


    }
}
