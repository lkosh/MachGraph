#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cassert>
#include <math.h>
#include "Utility.h"
GLuint texture;
using namespace std;
GLuint grass_texture;
const uint GRASS_INSTANCES = 5000	; // Количество травинок

GL::Camera camera;               // Мы предоставляем Вам реализацию камеры. В OpenGL камера - это просто 2 матрицы. Модельно-видовая матрица и матрица проекции. // ###
                                 // Задача этого класса только в том чтобы обработать ввод с клавиатуры и правильно сформировать эти матрицы.
                                 // Вы можете просто пользоваться этим классом для расчёта указанных матриц.


float koef_elast = 10;
GLuint grassPointsCount; // Количество вершин у модели травинки
GLuint grassShader;      // Шейдер, рисующий траву
GLuint grassVAO;         // VAO для травы (что такое VAO почитайте в доках)
GLuint grassVariance;    // Буфер для смещения координат травинок
GLuint grassAngle;
GLuint grassRot;
vector<VM::vec4> grassVarianceData(GRASS_INSTANCES); // Вектор со смещениями для координат травинок
vector <VM::vec2>velocity (GRASS_INSTANCES);
 
int Smooth = 1;
int WIND_ON = 1;//flag turn on/off wind

GLuint groundShader; // Шейдер для земли
GLuint groundVAO; // VAO для земли
int priv_time = 0;
float mass = 1;
float delta_time = 0;
float k = 3e-5;
double l  =0.001;
float wind_max_x  = 0.005, wind_max_y = 0.005;
//VM::vec2 wind_start(0,0);
vector <VM::vec2>wind_start (GRASS_INSTANCES);
VM::vec2 wind_force (1e-4,1e-4);
// Размеры экрана
uint screenWidth = 800;
uint screenHeight = 600;


//Углы между полигонами травинки 
vector <float> grassAngleData(GRASS_INSTANCES);
vector <float> grassRotData(GRASS_INSTANCES);
// Это для захвата мышки. Вам это не потребуется (это не значит, что нужно удалять эту строку)
bool captureMouse = true;

// Функция, рисующая замлю
void DrawGround() {
    // Используем шейдер для земли
    glUseProgram(groundShader);                                                  CHECK_GL_ERRORS

    // Устанавливаем юниформ для шейдера. В данном случае передадим перспективную матрицу камеры
    // Находим локацию юниформа 'camera' в шейдере
    GLint cameraLocation = glGetUniformLocation(groundShader, "camera");         CHECK_GL_ERRORS
    // Устанавливаем юниформ (загружаем на GPU матрицу проекции?)                                                     // ###
    glUniformMatrix4fv(cameraLocation, 1, GL_TRUE, camera.getMatrix().data().data()); CHECK_GL_ERRORS
    
    //lightning : passing viewer position to ground.frag
	GLint viewPosLoc = glGetUniformLocation(groundShader, "viewPos");					CHECK_GL_ERRORS
	glUniform3f(viewPosLoc, camera.position.x, camera.position.y, camera.position.z);	CHECK_GL_ERRORS
	
	//liggting : passing light attributes to shader
	GLint lightAmbientLoc = glGetUniformLocation(groundShader, "light.ambient");		CHECK_GL_ERRORS
	GLint lightDiffuseLoc = glGetUniformLocation(groundShader, "light.diffuse");		CHECK_GL_ERRORS
	GLint lightSpecularLoc = glGetUniformLocation(groundShader, "light.specular");		CHECK_GL_ERRORS
	glUniform3f(lightAmbientLoc, 0.2f, 0.2f, 0.2f);										CHECK_GL_ERRORS	
	glUniform3f(lightDiffuseLoc, 1.0f, 1.0f, 1.0f); // Let’s darken the light a bit to fit the scene
	glUniform3f(lightSpecularLoc, 1.0f, 1.0f, 1.0f);									CHECK_GL_ERRORS
	
	
	//lighting: passing material proprties to shader 
    GLint matAmbientLoc = glGetUniformLocation(groundShader, "material.ambient");		CHECK_GL_ERRORS
	//GLint matDiffuseLoc = glGetUniformLocation(groundShader, "material.diffuse");		CHECK_GL_ERRORS
	GLint matSpecularLoc = glGetUniformLocation(groundShader, "material.specular");		CHECK_GL_ERRORS
	GLint matShineLoc= glGetUniformLocation(groundShader, "material.shininess");		CHECK_GL_ERRORS
	
	glUniform3f(matAmbientLoc, 1.0f, 0.5f, 0.31f);										CHECK_GL_ERRORS	
	//glUniform3f(matDiffuseLoc, 1.0f, 0.5f, 0.31f);										CHECK_GL_ERRORS
	glUniform3f(matSpecularLoc, 0.5f, 0.5f, 0.5f);										CHECK_GL_ERRORS
	glUniform1f(matShineLoc,32.0f);														CHECK_GL_ERRORS
	glUniform1i(glGetUniformLocation(groundShader, "material.diffuse"), 0);
	
	GL::bindTexture(groundShader, "ourTexture", texture, 0);						CHECK_GL_ERRORS
	
    // Подключаем VAO, который содержит буферы, необходимые для отрисовки земли
    glBindVertexArray(groundVAO);                                                CHECK_GL_ERRORS

    // Рисуем землю: 2 треугольника (6 вершин)
    glDrawArrays(GL_TRIANGLES, 0, 6);                                            CHECK_GL_ERRORS

    // Отсоединяем VAO
    glBindVertexArray(0);                                                        CHECK_GL_ERRORS
    // Отключаем шейдер
    glUseProgram(0);                                                             CHECK_GL_ERRORS
}
namespace VM
{
    VM::vec2 operator*(float a, vec2 v)
    {
        return VM::vec2(a, a) * v;
    }
}


//Initialising light
void init()
{
    glClearColor(0, 0.2, 0.8, 1.0);												     CHECK_GL_ERRORS
    glEnable(GL_LIGHTING);															     CHECK_GL_ERRORS
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);								     CHECK_GL_ERRORS
    glEnable(GL_NORMALIZE);																     CHECK_GL_ERRORS
}
void init_l()
{
    float light0_diffuse[] = {0.4, 0.7, 0.2}; 								     		
    float light0_direction[] = {0.0, -1.0, -1.0, 0.0}; 							     	
	float light0_ambient[] = {1.0,1.0,1.0,1.0};											
    glEnable(GL_LIGHT0); 															 CHECK_GL_ERRORS

    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse); 									     CHECK_GL_ERRORS
    glLightfv(GL_LIGHT0, GL_POSITION, light0_direction); 										     CHECK_GL_ERRORS
    glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);									     CHECK_GL_ERRORS
}

// Обновление смещения травинок
void UpdateGrassVariance() {
    // Генерация случайных смещений
    float new_time = glutGet(GLUT_ELAPSED_TIME);
    float w0 = sqrt(k/mass);
    float w = 0.003;
    
	
   
	// Chaging alphas with respect to wind force
	
 
    if (priv_time){
		//wind_force = VM::vec2(0,0);
		delta_time = new_time - priv_time;
	}
	else{

		priv_time = glutGet(GLUT_ELAPSED_TIME);
	}
	
	//	wind_force = VM::vec2(0.001,0.001);
	//wind_force = VM::vec2(wind_max_x * cos(priv_time * w*1.0) + wind_max_x ,
						//wind_max_y * cos(priv_time * w*1.0 ) + wind_max_y);
			

    VM::vec2 F_gook , F_tr;
    VM::vec2 delta_var;
    VM::vec2 F_SUM; 
    VM::vec2 accel (0,0);
    int fl = 1; //DEBUG
   // wind_force = VM::vec2(0.001,0.001);
  		
    if (!WIND_ON)
		wind_force = VM::vec2(0,0);
    for (uint i = 0; i < GRASS_INSTANCES; ++i) {
		//wind_force += VM::vec2(wind_start[i].x,  wind_start[i].y);
	 wind_force = VM::vec2(wind_max_x * cos(priv_time * w*1.0)*cos(wind_start[i].x) + 2*wind_max_x ,
						wind_max_y * cos(priv_time * w*1.0 + wind_start[i].x) + wind_max_y);
		
	//	cout<<"wind force: "<<wind_force<<endl;		
		
		//VM::vec2 prev_speed = velocity[i];
        //VM::vec4 prev_grassVariance = grassVarianceData[i];
 
        //F_gook.x = -k * prev_grassVariance.x;
        //F_gook.y = -k * prev_grassVariance.z;
 
   
   
		
 
        //delta_var = 0.1*prev_speed*delta_time;
        
 
        //F_tr.x = - l * prev_speed.x*prev_speed.x;
        //F_tr.y = - l * prev_speed.y*prev_speed.y;
		//F_SUM = wind_force +F_gook ;
		////F0 = wind_force omega = 0
		//if (fl){
			//cout<<"gook: "<<F_gook<<endl;
			//cout<<"tr: "<<F_tr<<endl;
			//cout<<"delta: "<<delta_var<<endl;
			//fl=0;
		//}
		
		
		//accel = (F_gook + wind_force)/mass;
        //velocity[i] = accel * priv_time * 0.1;
        //grassVarianceData[i] += VM::vec4(delta_var.x, 0, delta_var.y, 0) ;
        
        grassVarianceData[i] += VM::vec4(0,0,0,0);
        //if (grassVarianceData[i].x > 50)
			//grassVarianceData[i].x = 50;
		//if (grassVarianceData[i].y > 50)
			//grassVarianceData[i].y = 50;	
			
				
		grassAngleData[i] = abs(sin(100*(wind_force.x ) ));
        //std::cout << grassAngleData[i] << std::endl;
    }
    
    priv_time += delta_time;
	 glBindBuffer(GL_ARRAY_BUFFER, grassAngle);   								  CHECK_GL_ERRORS
	 glBufferData(GL_ARRAY_BUFFER, sizeof(float) * GRASS_INSTANCES, grassAngleData.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS
    // Отвязываем буфер
   // glBindBuffer(GL_ARRAY_BUFFER, 0);                                            CHECK_GL_ERRORS
    
  
    
    // Привязываем буфер, содержащий смещения
    glBindBuffer(GL_ARRAY_BUFFER, grassVariance);                                CHECK_GL_ERRORS
    // Загружаем данные в видеопамять
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * GRASS_INSTANCES, grassVarianceData.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS
    // Отвязываем буфер
    glBindBuffer(GL_ARRAY_BUFFER, 0);                                            CHECK_GL_ERRORS
}


// Рисование травы
void DrawGrass() {
    // Тут то же самое, что и в рисовании земли
    glUseProgram(grassShader);                                                   CHECK_GL_ERRORS
    GLint cameraLocation = glGetUniformLocation(grassShader, "camera");          CHECK_GL_ERRORS
    glUniformMatrix4fv(cameraLocation, 1, GL_TRUE, camera.getMatrix().data().data()); CHECK_GL_ERRORS
    
    GL::bindTexture(grassShader, "grassTexture", grass_texture, 1);				CHECK_GL_ERRORS

    glBindVertexArray(grassVAO);                                                 CHECK_GL_ERRORS
    // Обновляем смещения для травы
    UpdateGrassVariance();
    // Отрисовка травинок в количестве GRASS_INSTANCES
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, grassPointsCount, GRASS_INSTANCES);   CHECK_GL_ERRORS
    glBindVertexArray(0);                                                        CHECK_GL_ERRORS
    glUseProgram(0);                                                             CHECK_GL_ERRORS
}

// Эта функция вызывается для обновления экрана
void RenderLayouts() {
    // Включение буфера глубины
    glEnable(GL_DEPTH_TEST);
    // Очистка буфера глубины и цветового буфера
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   //light
    init_l();
    // Рисуем меши
    DrawGround();
    DrawGrass();
  //  glDisable(GL_LIGHT0);
    glutSwapBuffers();
}

// Завершение программы
void FinishProgram() {
    glutDestroyWindow(glutGetWindow());
}

// Обработка события нажатия клавиши (специальные клавиши обрабатываются в функции SpecialButtons)
void KeyboardEvents(unsigned char key, int x, int y) {
    if (key == 27) {
        FinishProgram();
    } else if (key == 'w') {
        camera.goForward();
    } else if (key == 's') {
        camera.goBack();
    } else if (key == 'm') {
        captureMouse = !captureMouse;
        if (captureMouse) {
            glutWarpPointer(screenWidth / 2, screenHeight / 2);
            glutSetCursor(GLUT_CURSOR_NONE);
        } else {
            glutSetCursor(GLUT_CURSOR_RIGHT_ARROW);
        }
    }
    else if (key =='a'){
		if (!Smooth)
			glEnable(GL_MULTISAMPLE);
		else 
			glDisable(GL_MULTISAMPLE); 
		Smooth = !Smooth;
	}
    else if (key == 'p'){
		WIND_ON = !WIND_ON;
	}
}

// Обработка события нажатия специальных клавиш
void SpecialButtons(int key, int x, int y) {
    if (key == GLUT_KEY_RIGHT) {
        camera.rotateY(0.02);
    } else if (key == GLUT_KEY_LEFT) {
        camera.rotateY(-0.02);
    } else if (key == GLUT_KEY_UP) {
        camera.rotateTop(-0.02);
    } else if (key == GLUT_KEY_DOWN) {
        camera.rotateTop(0.02);
    }
}

void IdleFunc() {
    glutPostRedisplay();
}

// Обработка события движения мыши
void MouseMove(int x, int y) {
    if (captureMouse) {
        int centerX = screenWidth / 2,
            centerY = screenHeight / 2;
        if (x != centerX || y != centerY) {
            camera.rotateY((x - centerX) / 1000.0f);
            camera.rotateTop((y - centerY) / 1000.0f);
            glutWarpPointer(centerX, centerY);
        }
    }
}

// Обработка нажатия кнопки мыши
void MouseClick(int button, int state, int x, int y) {
}

// Событие изменение размера окна
void windowReshapeFunc(GLint newWidth, GLint newHeight) {
    glViewport(0, 0, newWidth, newHeight);
    screenWidth = newWidth;
    screenHeight = newHeight;

    camera.screenRatio = (float)screenWidth / screenHeight;
}

// Инициализация окна
void InitializeGLUT(int argc, char **argv) {
    glutInit(&argc, argv);
    glutSetOption(GLUT_MULTISAMPLE, 8);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitContextVersion(3, 0);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutInitWindowPosition(-1, -1);
    glutInitWindowSize(screenWidth, screenHeight);
    glutCreateWindow("Computer Graphics 3");
    glutWarpPointer(400, 300);
    glutSetCursor(GLUT_CURSOR_NONE);
// init light
	init();

    glutDisplayFunc(RenderLayouts);
    glutKeyboardFunc(KeyboardEvents);
    glutSpecialFunc(SpecialButtons);
    glutIdleFunc(IdleFunc);
    glutPassiveMotionFunc(MouseMove);
    glutMouseFunc(MouseClick);
    glutReshapeFunc(windowReshapeFunc);
}

// Генерация позиций травинок (эту функцию вам придётся переписать)
vector<VM::vec2> GenerateGrassPositions() {
	vector<VM::vec2> grassPositions(GRASS_INSTANCES);
    for (uint i = 0; i < GRASS_INSTANCES; ++i) {
        //grassPositions[i] = VM::vec2((i % 4) / 4.0, (i / 4) / 4.0) + VM::vec2(1, 1) / 8 ;//+ VM::vec2(rand()/RAND_MAX);
		//grassPositions[i] = 0.01*i;
		grassPositions[i] =  /*VM::vec2(0.5, 0.5);*/VM::vec2((double)rand()/RAND_MAX , (double)rand()/RAND_MAX);
		}
    return grassPositions;
}

// Здесь вам нужно будет генерировать меш
vector<VM::vec4> GenMesh(uint n) {
    return {//5 triangles
        VM::vec4(0, 0, 0, 1),//0
        VM::vec4(1, 0, 0, 1),//1
        
        VM::vec4(0.1, 0.2, 0, 1),
        VM::vec4(0.9, 0.2, 0, 1),
        
		VM::vec4(0.2, 0.4, 0, 1),
		VM::vec4(0.9, 0.4, 0, 1),
		
		VM::vec4(0.3, 0.6, 0, 1),
		VM::vec4(0.7, 0.6, 0, 1),
		
		VM::vec4(0.4,0.8, 0, 1),
		VM::vec4(0.6, 0.8, 0, 1),
		
		VM::vec4(0.5, 1, 0, 1),
		 //VM::vec4(0, 0, 0, 1),
        //VM::vec4(0, 0, 0.005, 1),
        //VM::vec4(0, 0.2, 0.001, 1),
        //VM::vec4(0, 0.2, 0.004, 1),
        //VM::vec4(0.1, 0.4, 0.002, 1),
        //VM::vec4(0.1, 0.4, 0.003, 1),
        //VM::vec4(0.4, 0.7, 0.0025, 1)
    };
}
// 
// Создание травы
void CreateGrass() {
	
    uint LOD = 1;
    // Создаём меш
    vector<VM::vec4> grassPoints = GenMesh(LOD);
    // Сохраняем количество вершин в меше травы
    grassPointsCount = grassPoints.size();
    cout<<grassPointsCount<<endl;
    // Создаём позиции для травинок
    vector<VM::vec2> grassPositions = GenerateGrassPositions();
    // Инициализация смещений для травинок
    for (uint i = 0; i < GRASS_INSTANCES; ++i) {
		
		//float y = rand()/RAND_MAX;
        grassVarianceData[i] = VM::vec4(1, 1, 1, 1	);
        grassAngleData[i] = 0	;
        wind_start[i] = VM::vec2(wind_max_x* cos(1.6*grassPositions[i].x), wind_max_y*cos(1.6*grassPositions[i].y)); 
        cout<< grassPoints[i]<<"  "<<wind_start[i]<<endl;
        if (i % 5 == 0)
			grassRotData[i] =1.6*((float)rand()/RAND_MAX - .5);
		else
			grassRotData[i] = 0;
        cout<<grassRotData[i]<<endl;
    }

    /* Компилируем шейдеры
    Эта функция принимает на вход название шейдера 'shaderName',
    читает файлы shaders/{shaderName}.vert - вершинный шейдер
    и shaders/{shaderName}.frag - фрагментный шейдер,
    компилирует их и линкует.
    */
	cout<<"grass"<<endl;
    grassShader = GL::CompileShaderProgram("grass");

    // Здесь создаём буфер
    GLuint pointsBuffer;
    // Это генерация одного буфера (в pointsBuffer хранится идентификатор буфера)
    glGenBuffers(1, &pointsBuffer);                                              CHECK_GL_ERRORS
    // Привязываем сгенерированный буфер
    glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);                                 CHECK_GL_ERRORS
    // Заполняем буфер данными из вектора
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * grassPoints.size(), grassPoints.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    // Создание VAO
    // Генерация VAO
    glGenVertexArrays(1, &grassVAO);                                             CHECK_GL_ERRORS
    // Привязка VAO
    glBindVertexArray(grassVAO);                                                 CHECK_GL_ERRORS

    // Получение локации параметра 'point' в шейдере
    GLuint pointsLocation = glGetAttribLocation(grassShader, "point");           CHECK_GL_ERRORS
    // Подключаем массив атрибутов к данной локации
    glEnableVertexAttribArray(pointsLocation);                                   CHECK_GL_ERRORS
    // Устанавливаем параметры для получения данных из массива (по 4 значение типа float на одну вершину)
    glVertexAttribPointer(pointsLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);          CHECK_GL_ERRORS

    // Создаём буфер для позиций травинок
    GLuint positionBuffer;
    glGenBuffers(1, &positionBuffer);                                            CHECK_GL_ERRORS
    // Здесь мы привязываем новый буфер, так что дальше вся работа будет с ним до следующего вызова glBindBuffer
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);                               CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec2) * grassPositions.size(), grassPositions.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    GLuint positionLocation = glGetAttribLocation(grassShader, "position");      CHECK_GL_ERRORS
    glEnableVertexAttribArray(positionLocation);                                 CHECK_GL_ERRORS
    glVertexAttribPointer(positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    // Здесь мы указываем, что нужно брать новое значение из этого буфера для каждого инстанса (для каждой травинки)
    glVertexAttribDivisor(positionLocation, 1);                                  CHECK_GL_ERRORS

    // Создаём буфер для смещения травинок
    glGenBuffers(1, &grassVariance);                                            CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, grassVariance);                               CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * GRASS_INSTANCES, grassVarianceData.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    GLuint varianceLocation = glGetAttribLocation(grassShader, "variance");      CHECK_GL_ERRORS
    glEnableVertexAttribArray(varianceLocation);                                 CHECK_GL_ERRORS
    glVertexAttribPointer(varianceLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    glVertexAttribDivisor(varianceLocation, 1);                                  CHECK_GL_ERRORS
    
    
    GLuint angleLocation;
	//ANGLE
	glGenBuffers(1, &grassAngle);                                            CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, grassAngle);                               CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * GRASS_INSTANCES, grassAngleData.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    angleLocation = glGetAttribLocation(grassShader, "angle");      CHECK_GL_ERRORS
    glEnableVertexAttribArray(angleLocation);                                 CHECK_GL_ERRORS
    glVertexAttribPointer(angleLocation, 1, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    glVertexAttribDivisor(angleLocation, 1);                                  CHECK_GL_ERRORS
    //ROTATION

	glGenBuffers(1, &grassRot);                                            CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, grassRot);                               CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * GRASS_INSTANCES, grassRotData.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    GLuint rotLocation = glGetAttribLocation(grassShader, "rot");      CHECK_GL_ERRORS
    glEnableVertexAttribArray(rotLocation);                                 CHECK_GL_ERRORS
    glVertexAttribPointer(rotLocation, 1, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    glVertexAttribDivisor(rotLocation, 1);                                  CHECK_GL_ERRORS

    // Отвязываем VAO
    glBindVertexArray(0);                                                        CHECK_GL_ERRORS
    // Отвязываем буфер
    glBindBuffer(GL_ARRAY_BUFFER, 0);                                            CHECK_GL_ERRORS
	
	cout<<"begin"<<endl;
    int width, height;
    unsigned char* image = SOIL_load_image("../grass.jpg", &width, &height, 0 , SOIL_LOAD_RGB);  			CHECK_GL_ERRORS
	assert(image != nullptr);
	glGenTextures(1, &grass_texture);																		CHECK_GL_ERRORS

    glBindTexture(GL_TEXTURE_2D, grass_texture);																CHECK_GL_ERRORS
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);										CHECK_GL_ERRORS
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);										CHECK_GL_ERRORS
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);									CHECK_GL_ERRORS
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);									CHECK_GL_ERRORS
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);						CHECK_GL_ERRORS
	
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);          CHECK_GL_ERRORS

	glGenerateMipmap(GL_TEXTURE_2D);																	CHECK_GL_ERRORS

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);									CHECK_GL_ERRORS    			
   
	cout<<"end"<<endl;

    
   glBindTexture(GL_TEXTURE_2D, 0);
	SOIL_free_image_data(image);





}

// Создаём камеру (Если шаблонная камера вам не нравится, то можете переделать, но я бы не стал)
void CreateCamera() {
    camera.angle = 45.0f / 180.0f * M_PI;
    camera.direction = VM::vec3(0, 0.3, -1);
    camera.position = VM::vec3(0.5, 0.2, 0);
    camera.screenRatio = (float)screenWidth / screenHeight;
    camera.up = VM::vec3(0, 1, 0);
    camera.zfar = 50.0f;
    camera.znear = 0.05f;
}

// Создаём замлю
void CreateGround() {
    // Земля состоит из двух треугольников
    vector<VM::vec4> meshPoints = {
        VM::vec4(0, 0, 0, 1),
        VM::vec4(1, 0, 0, 1),
        VM::vec4(1, 0, 1, 1),
        VM::vec4(0, 0, 0, 1),
        VM::vec4(1, 0, 1, 1),
        VM::vec4(0, 0, 1, 1),
    };
	//GLfloat texCoords[] = {
		//0.0f, 0.0f,
		//1.0f, 0.0f,
		//0.5f, 1.0f,
		
		//0.0f, 0.0f,
		//0.5f, 1.0f,
		//1.0f, 0.0f
	//}
    // Подробнее о том, как это работает читайте в функции CreateGrass
	// ЗДЕСЬ БЫЛИ ФУНКЦИИ ТЕКСТУР, НО ОНИ ПЛОХО СЕБЯ ВЕЛИ! И ТАК БУДЕТ С КАЖДЫМ!
    
    groundShader = GL::CompileShaderProgram("ground");													CHECK_GL_ERRORS



    GLuint pointsBuffer;
    glGenBuffers(1, &pointsBuffer);                                              CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);                                 CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * meshPoints.size(), meshPoints.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    glGenVertexArrays(1, &groundVAO);                                            CHECK_GL_ERRORS
    glBindVertexArray(groundVAO);                                                CHECK_GL_ERRORS

    GLuint index = glGetAttribLocation(groundShader, "point");                   CHECK_GL_ERRORS
    glEnableVertexAttribArray(index);                                            CHECK_GL_ERRORS
    glVertexAttribPointer(index, 4, GL_FLOAT, GL_FALSE, 0, 0);                   CHECK_GL_ERRORS
	glBindVertexArray(0);                                                        CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, 0);                                            CHECK_GL_ERRORS
    
	
    int width, height;
    unsigned char* image = SOIL_load_image("../ground.bmp", &width, &height, 0 , SOIL_LOAD_RGB);  			CHECK_GL_ERRORS
	assert(image != nullptr);
	glGenTextures(1, &texture);																		CHECK_GL_ERRORS

    glBindTexture(GL_TEXTURE_2D, texture);																CHECK_GL_ERRORS
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);										CHECK_GL_ERRORS
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);										CHECK_GL_ERRORS
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);									CHECK_GL_ERRORS
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);									CHECK_GL_ERRORS
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);						CHECK_GL_ERRORS
	
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);          CHECK_GL_ERRORS

	glGenerateMipmap(GL_TEXTURE_2D);																	CHECK_GL_ERRORS

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);									CHECK_GL_ERRORS    			
   


    
   glBindTexture(GL_TEXTURE_2D, 0);
	SOIL_free_image_data(image);
}

int main(int argc, char **argv)
{
    putenv("MESA_GL_VERSION_OVERRIDE=3.3COMPAT");
    try {
        cout << "Start" << endl;
        InitializeGLUT(argc, argv);
        cout << "GLUT inited" << endl;
        glewInit();
        cout << "glew inited" << endl;
        CreateCamera();
        cout << "Camera created" << endl;
        
		//	init();// initialise light
        CreateGrass();
        cout << "Grass created" << endl;
        CreateGround();
        cout << "Ground created" << endl;
       
       
        glutMainLoop();
    } catch (string s) {
        cout << s << endl;
    }
}
