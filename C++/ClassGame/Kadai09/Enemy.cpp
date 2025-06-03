/**
* @file Enemy.cpp
* @brief 課題9 : Enemyクラスメンバー関数
* @author IT11H308 ok230112 江藤大晴
* @date 2023/12/1
*/

#include <iostream>
using namespace std;

#include "Enemy.h"

/* 名前を設定 */
void Enemy::setName(string na) {
	name = na;
}

/* 名前を取得 */
string Enemy::getName() {
	return name;
}

/* HPを設定 */
void Enemy::setHitPoint(int hp) {
	hitPoint = hp;
}

/* HPを取得 */
int Enemy::getHitPoint() {
	return hitPoint;
}

/* 状態を表示 */
void Enemy::showData() {

	/* 名前と現在のHPを表示 */
	cout << getName() << " HP=" << getHitPoint() << endl;
}

/* ダメージを受ける */
void Enemy::receiveDamage(int damage) {

	/* ダメージの表示 */
	cout << getName() << "は" << damage << "のダメージを受けた！" << endl;

	/* HPをdamage分引く */
	hitPoint -= damage;
	
	/* HPが0以下なら「倒れた」*/
	if (hitPoint <= 0) {
		hitPoint = 0;
		cout << getName() << "は倒れた！" << endl;
	}
}