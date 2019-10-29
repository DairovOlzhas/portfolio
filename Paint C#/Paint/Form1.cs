using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Paint
{
    public enum Tool
    {
        Pencil,
        Line,
        Test,
        Rectangle,
        Circle,
        Triangler,
        Trianglep,
        Triangleb,
        Cub,
        CubInCub,
        Erase,
        Fill
    }

    public partial class Form1 : Form
    {
        Pen pen;
        Pen erase;
        Graphics g;
        GraphicsPath path;
        Bitmap bmp;
        Point prev, cur;
        Queue<Point> q = new Queue<Point>();

        Color init_color, fill_color;

        Tool t;

        public Form1()
        {
            InitializeComponent();
            path = new GraphicsPath();
            pen = new Pen(Color.Black, 3);
            erase = new Pen(Color.White, 5);
            bmp = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            pictureBox1.Image = bmp;
            g = Graphics.FromImage(bmp);
            g.Clear(Color.White);
            t = Tool.Pencil;

        }

        private void ColorChange_Click(object sender, EventArgs e)
        {
            colorDialog1.ShowDialog();
            pen = new Pen(colorDialog1.Color, pen.Width);
            button1.BackColor = colorDialog1.Color;
        }

        private void pictureBox1_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.DrawPath(pen, path);
        }

        private void check(int x, int y)
        {
            if (x < 0 || x > bmp.Width - 1 || y < 0 || y > bmp.Height - 1)
                return;
            if (bmp.GetPixel(x, y) == init_color)
            {
                q.Enqueue(new Point(x, y));
                bmp.SetPixel(x, y, fill_color);
            }
        }

        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            prev = e.Location;
            if(t == Tool.Fill && bmp.GetPixel(e.X, e.Y).ToArgb() != pen.Color.ToArgb())
            {
                Color color = bmp.GetPixel(e.X, e.Y);
                //MessageBox.Show(color.ToArgb() + " and " + pen.Color.ToArgb());
                int x = e.Location.X;
                int y = e.Location.Y;
                fill_color = pen.Color;
                init_color = bmp.GetPixel(x, y); 
                q.Enqueue(new Point(x, y));
                bmp.SetPixel(x, y, fill_color);

                while(q.Count != 0)
                {
                    Point p = q.Dequeue();
                    check(p.X, p.Y - 1);
                    check(p.X + 1, p.Y);
                    check(p.X, p.Y + 1);
                    check(p.X - 1, p.Y);
                    //Thread.Sleep(1);
                }
                pictureBox1.Refresh();
            }
        }

        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            path.Reset();
            if(e.Button == MouseButtons.Left)
            {
                cur = e.Location;
                Point position = new Point();
                Size size = new Size(Math.Abs(cur.X - prev.X), Math.Abs(cur.Y - prev.Y));
                if (prev.X < cur.X && prev.Y < cur.Y)
                    position = new Point(prev.X, prev.Y);
                if (prev.X < cur.X && prev.Y > cur.Y)
                    position = new Point(prev.X, cur.Y);
                if (prev.X > cur.X && prev.Y > cur.Y)
                    position = new Point(cur.X, cur.Y);
                if (prev.X > cur.X && prev.Y < cur.Y)
                    position = new Point(cur.X, prev.Y);

                

                switch (t)
                {
                    case Tool.Test:
                        List<GraphicsPath> paths = new List<GraphicsPath>();
                        paths.Add(new GraphicsPath());
                        paths.Add(new GraphicsPath());
                        paths[1].AddLine(prev, new Point(prev.X + 20, prev.Y + 20));
                        paths[0].AddLine(new Point(prev.X + 20 , prev.Y - 20), cur);
                        Point[] test =
                        {
                            prev,
                            new Point(prev.X + cur.Y, prev.Y + cur.X),
                            cur
                        };
                        path.AddPath(paths[1], false);
                        path.AddPath(paths[0], false);
                        break;

                    case Tool.Pencil:
                        path.AddLine(prev, cur);
                        g.DrawPath(pen, path);
                        prev = cur;
                        break;

                    case Tool.Line:
                        path.AddLine(prev, cur);
                        break;
                    case Tool.Rectangle:
                        path.AddRectangle(new Rectangle(position, size));
                        break;
                    case Tool.Circle:
                        path.AddEllipse(new Rectangle(position, size));
                        break;
                    case Tool.Cub:  
                        path.AddRectangle(new Rectangle(position, new Size(size.Width / 3 * 2, size.Height / 3 * 2)));
                        path.AddRectangle(new Rectangle(new Point(position.X + size.Width / 3, position.Y + size.Height / 3), new Size(size.Width / 3 * 2, size.Height / 3 * 2)));
                        path.AddLine(position, new Point(position.X + size.Width / 3, position.Y + size.Height / 3));
                        path.AddLine(new Point(position.X + size.Width, position.Y + size.Height/3), new Point(position.X + size.Width / 3 * 2, position.Y));
                        path.AddLine(new Point(position.X + size.Width / 3 * 2, position.Y + size.Height / 3 * 2), new Point(position.X + size.Width, position.Y + size.Height));
                        path.AddLine(new Point(position.X + size.Width/3, position.Y + size.Height), new Point(position.X, position.Y + size.Height / 3 * 2));
                        break;

                    case Tool.CubInCub:

                        List<GraphicsPath> pathcub = new List<GraphicsPath>();
                        for (int i = 0; i < 15; i++)
                            pathcub.Add(new GraphicsPath());
                        path.AddRectangle(new Rectangle(position, new Size(size.Width / 4 * 3, size.Height / 4 * 3)));
                        path.AddRectangle(new Rectangle(new Point(position.X + size.Width / 4, position.Y + size.Height / 4), new Size(size.Width / 4 * 3, size.Height / 4 * 3)));
                        pathcub[0].AddLine(position, new Point(position.X + size.Width / 4, position.Y + size.Height / 4));
                        pathcub[1].AddLine(new Point(position.X + size.Width, position.Y + size.Height / 4), new Point(position.X + size.Width / 4 * 3, position.Y));
                        pathcub[2].AddLine(new Point(position.X + size.Width / 4 * 3, position.Y + size.Height / 4 * 3), new Point(position.X + size.Width, position.Y + size.Height));
                        pathcub[3].AddLine(new Point(position.X + size.Width / 4, position.Y + size.Height), new Point(position.X, position.Y + size.Height / 4 * 3));

                        Point position1 = new Point(position.X + size.Width / 3, position.Y + size.Height / 3);
                        Size size1 = new Size(size.Width / 3, size.Height / 3);

                        path.AddRectangle(new Rectangle(position1, new Size(size1.Width / 4 * 3, size1.Height / 4 * 3)));
                        path.AddRectangle(new Rectangle(new Point(position1.X + size1.Width / 4, position1.Y + size1.Height / 4), new Size(size1.Width / 4 * 3, size1.Height / 4 * 3)));
                        pathcub[4].AddLine(position1, new Point(position1.X + size1.Width / 4, position1.Y + size1.Height / 4));
                        pathcub[5].AddLine(new Point(position1.X + size1.Width, position1.Y + size1.Height / 4), new Point(position1.X + size1.Width / 4 * 3, position1.Y));
                        pathcub[6].AddLine(new Point(position1.X + size1.Width / 4 * 3, position1.Y + size1.Height / 4 * 3), new Point(position1.X + size1.Width, position1.Y + size1.Height));
                        pathcub[7].AddLine(new Point(position1.X + size1.Width / 4, position1.Y + size1.Height), new Point(position1.X, position1.Y + size1.Height / 4 * 3));

                        pathcub[8].AddLine(new Point(position1.X, position1.Y + size1.Height / 4 * 3), new Point(position.X, position.Y + size.Height / 4 * 3));
                        pathcub[9].AddLine(new Point(position.X + size.Width / 4 * 3, position.Y + size.Height / 4 * 3), new Point(position1.X + size1.Width / 4 * 3, position1.Y + size1.Height / 4 * 3));
                        pathcub[10].AddLine(new Point(position1.X + size1.Width / 4 * 3, position1.Y), new Point(position.X + size.Width / 4 * 3, position.Y));
                        pathcub[11].AddLine(new Point(position.X + size.Width, position.Y + size.Height / 4), new Point(position1.X + size1.Width, position1.Y + size1.Height / 4));
                        pathcub[12].AddLine(new Point(position1.X + size1.Width / 4, position1.Y + size1.Height / 4), new Point(position.X + size.Width / 4, position.Y + size.Height / 4));
                        pathcub[13].AddLine(new Point(position.X + size.Width / 4, position.Y + size.Height), new Point(position1.X + size1.Width / 4, position1.Y + size1.Height));
                        for (int i = 0; i < 14; i++)
                            path.AddPath(pathcub[i], false);
                        break;

                    case Tool.Triangleb:
                        Point[] p =
                        {
                            new Point(position.X, position.Y + size.Height),
                            new Point(position.X + size.Width / 2, position.Y),
                            new Point(position.X + size.Width, position.Y + size.Height)
                        };
                        path.AddPolygon(p);
                        break;

                    case Tool.Trianglep:
                        Point[] p1 =
                        {
                            new Point(position.X, position.Y + size.Height),
                            new Point(position.X + size.Width, position.Y),
                            new Point(position.X + size.Width, position.Y + size.Height)
                        };
                        path.AddPolygon(p1);
                        break;

                    case Tool.Triangler:
                        int h = Convert.ToInt32(position.Y + Math.Sin(Math.PI / 3) * size.Width);
                        Point[] p2 =
                        {
                            new Point(position.X, h),
                            new Point(position.X + size.Width / 2, position.Y),
                            new Point(position.X + size.Width, h)
                        };
                        path.AddPolygon(p2);
                        break;
                    case Tool.Erase:
                        path.AddLine(prev, cur);
                        g.DrawPath(erase, path);
                        prev = cur;
                        break;
                }

                Refresh();
            }
        }

        private void pictureBox1_MouseUp(object sender, MouseEventArgs e)
        {
            g.DrawPath(pen, path);
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            switch (comboBox1.Text)
            {
                case "Pencil":
                    t = Tool.Pencil;
                    break;
                case "Test":
                    t = Tool.Test;
                    break;
                case "Line":
                    t = Tool.Line;
                    break;
                case "Trianglep":
                    t = Tool.Trianglep;
                    break;
                case "Triangler":
                    t = Tool.Triangler;
                    break;
                case "Triangleb":
                    t = Tool.Triangleb;
                    break;
                case "Circle":
                    t = Tool.Circle;
                    break;
                case "Rectangle":
                    t = Tool.Rectangle;
                    break;
                case "Cub":
                    t = Tool.Cub;
                    break;
                case "CubInCub":
                    t = Tool.CubInCub;
                    break;
                case "Erase":
                    t = Tool.Erase;
                    break;
            }
        }

        private void comboBox2_SelectedIndexChanged(object sender, EventArgs e)
        {
            pen = new Pen(pen.Color, int.Parse(comboBox2.Text));
        }
        private void comboBox3_SelectedIndexChanged(object sender, EventArgs e)
        {
            erase = new Pen(erase.Color, int.Parse(comboBox3.Text));
        }
        private void saveToolStripMenuItem_Click(object sender, EventArgs e)
        {

        }

        private void newFileToolStripMenuItem_Click(object sender, EventArgs e)
        {
            //bmp = new Bitmap(pictureBox1.Width, pictureBox1.Height);
           // g = Graphics.FromImage(bmp);
            g.Clear(Color.White);
            pictureBox1.Refresh();
        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            openFileDialog1.Filter = "Image Files(*.BMP;*.JPG;*.GIF)|*.BMP;*.JPG;*.GIF|All files (*.*)|*.*";
            if(openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                bmp = new Bitmap(Image.FromFile(openFileDialog1.FileName));
                pictureBox1.Image = bmp;
                g = Graphics.FromImage(bmp);
                Refresh();
            }
                
        }

        private void button2_Click(object sender, EventArgs e)
        {
            t = Tool.Erase;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            t = Tool.Fill;
        }
    }
}
