#include <iostream>
#include <iomanip>
#include <cmath>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

double norm(VectorXd v)
{
    return sqrt(v.dot(v));
}


int main()
{
    Vector2d xStar(-1.0, -1.0);

    MatrixXd err(2,3);

    Matrix2d A1 {
        {5.547001962252291e-01, -3.770900990025203e-02},
        {8.320502943378437e-01, -9.992887623566787e-01}
    };
    Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);
    Vector2d x_lu = A1.lu().solve(b1);
    err(0,0) = norm(x_lu-xStar)/norm(xStar);
    Vector2d x_qr = A1.householderQr().solve(b1);
    err(1,0) = norm(x_qr-xStar)/norm(xStar);

    Matrix2d A2 {
        {5.547001962252291e-01, -5.540607316466765e-01},
        {8.320502943378437e-01, -8.324762492991313e-01}
    };
    Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);
    x_lu = A2.lu().solve(b2);
    err(0,1) = norm(x_lu-xStar)/norm(xStar);
    x_qr = A2.householderQr().solve(b2);
    err(1,1) = norm(x_qr-xStar)/norm(xStar);

    Matrix2d A3 {
        {5.547001962252291e-01, -5.547001955851905e-01},
        {8.320502943378437e-01, -8.320502947645361e-01}
    };
    Vector2d b3(-6.400391328043042e-10, 4.266924591433963e-10);
    x_lu = A3.lu().solve(b3);
    err(0,2) = norm(x_lu-xStar)/norm(xStar);
    x_qr = A3.householderQr().solve(b3);
    err(1,2) = norm(x_qr-xStar)/norm(xStar);

    for (int i = 0; i<3; i++){
        cout << "The relative error for the problem " << i+1 << " solved with PALU is: " << scientific << err(0,i) <<endl;
        cout << "The relative error for the problem " << i+1 << " solved with QR is: " << scientific << err(1,i) <<endl;
    }

    return 0;
}
