using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace SnakeGame
{
    public class Game
    {
        public static bool check = true;
        public static string name = "New Game";
        public static ConsoleKeyInfo btn;
        public static int dx, dy;
        public static int sizex = 80, sizey = 30;
        public static Snake snake;
        public static Wall wall;
        public static Food food;
        public static int level;
        public static int score;
        public static int speed;



        public static void NewGame()
        {
            check = true;
            btn = new ConsoleKeyInfo();
            dx = 1; 
            dy = 0;
            level = 1;
            snake = new Snake(1);
            wall = new Wall();
            food = new Food();            
            speed = 100;
        }

        public static void GameProcess()
        {
            check = true;
            Game.Draw();
            while (check && btn.Key != ConsoleKey.Escape)
            {
                snake.Draw();
                food.Draw();
                DrawHeader();
                snake.Move(dx, dy);
                food.Eat();
                check = !GameOver();
                if(snake.body.Count - 5> score)
                {
                    level++;
                    wall.ReadWallLevel(level);
                    Console.Clear();
                    wall.Draw();
                    if(speed > 20)
                        speed -= 20;
                }
                Thread.Sleep(speed);
            }

            if (check)
            {
                SaveDataGame(false);
            }
            else
                ShowGameOver();
        }

        public static bool GameOver()
        {
            if (Collision(snake.body[0]))
            {
                Game.NewGame();
                return true;
            }
            return false;
        }
        public static void ShowGameOver()
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.SetCursorPosition(11, 12);
            Console.WriteLine("                                                         ");
            Console.SetCursorPosition(11, 13);
            Console.WriteLine(" #####   ###    # #   ####      ###   #   #  ####  ####  ");
            Console.SetCursorPosition(11, 14);
            Console.WriteLine(" #      #   #  # # #  #        #   #  #   #  #     #   # ");
            Console.SetCursorPosition(11, 15);
            Console.WriteLine(" #  ##  # # #  # # #  ####     #   #  #   #  ####  ####  ");
            Console.SetCursorPosition(11, 16);
            Console.WriteLine(" #   #  #   #  #   #  #        #   #   # #   #     #   # ");
            Console.SetCursorPosition(11, 17);
            Console.WriteLine(" #####  #   #  #   #  ####      ###     #    ####  #   # ");
            Console.SetCursorPosition(11, 18);
            Console.WriteLine("                                                         ");           
        }

        public static void ReadDataGame(string s)
        {
            string[] types = { "Snake", "Wall", "Food", "Direction", "Level" };
            foreach (string type in types)
            {
                string path = string.Format(@"Users/{0}/{1}.xml", s, type);

                FileStream fs = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite);

                XmlSerializer xs;
                if (type == "Snake")
                {
                    xs = new XmlSerializer(typeof(Snake));
                    Game.snake = xs.Deserialize(fs) as Snake;
                }
                if (type == "Wall")
                {
                    xs = new XmlSerializer(typeof(Wall));
                    Game.wall = xs.Deserialize(fs) as Wall;
                }
                if (type == "Food")
                {
                    xs = new XmlSerializer(typeof(Food));
                    Game.food = xs.Deserialize(fs) as Food;
                }
                if (type == "Direction")
                {
                    xs = new XmlSerializer(typeof(int[]));
                    int[] direction = xs.Deserialize(fs) as int[];
                    Game.dx = direction[0];
                    Game.dy = direction[1];
                }
                if (type == "Level")
                {
                    xs = new XmlSerializer(typeof(int[]));
                    Game.level = (xs.Deserialize(fs) as int[])[0];
                }

                fs.Close();
            }
                
        }
        public static void SaveDataGame(bool s)
        {
            string name;
            if (s)
            {
                Console.Clear();
                Console.SetCursorPosition(30, 11);
                Console.WriteLine("Only letters");
                Console.SetCursorPosition(30, 12);
                Console.WriteLine("Your Name:");
                Console.SetCursorPosition(41, 12);
                name = Console.ReadLine();
            }
            else
            {
                name = Game.name;

            }

            string[] types = {"Snake", "Wall", "Food", "Direction", "Level"};
            foreach(string type in types)
            {
                string path = string.Format(@"Users/{0}/{1}.xml", name, type);
                DirectoryInfo file = new DirectoryInfo(string.Format(@"Users/{0}", name));
                if (file.Exists)
                {
                    new FileInfo(path).Delete();

                }
                else
                    file.Create();
                    
                FileStream fs = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite);
                XmlSerializer xs;
                if (type == "Snake")
                {
                    xs = new XmlSerializer(typeof(Snake));
                    xs.Serialize(fs, Game.snake);                    
                }
                if (type == "Wall")
                {
                    xs = new XmlSerializer(typeof(Wall));
                    xs.Serialize(fs, Game.wall);
                }
                if (type == "Food")
                {
                    xs = new XmlSerializer(typeof(Food));
                    xs.Serialize(fs, Game.food);
                }
                if (type == "Direction")
                {
                    int[] direction = { Game.dx, Game.dy };
                    xs = new XmlSerializer(typeof(int[]));
                    xs.Serialize(fs, direction);
                }
                if (type == "Level")
                {
                    int[] level = { Game.level };
                    xs = new XmlSerializer(typeof(int[]));
                    xs.Serialize(fs, level);
                }
                fs.Close();
            }
        }

        public static bool Collision(Point point)
        {
            for(int i = 1; i < snake.body.Count - 1; i++)
                if (point.x == snake.body[i].x && point.y == snake.body[i].y)
                {
                    return true;
                }

          
            foreach (Point p in wall.body)
                if (point.x == p.x && point.y == p.y)
                {
                    return true;
                }
             
            return false;
        }

        public static void DrawHeader()
        {
            Console.SetCursorPosition(0, 0);
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("    Lvl " + level + "     Score: " + (Game.snake.body.Count - 4) + "  Speed:" + speed + "  " );
            Console.WriteLine("________________________________________________________________________________");
        }

        public static void Draw()
        {
            Game.wall.Draw();
            Game.food.Draw();
            Game.snake.Draw();
        }
    }

    


}
