/**
* @file Enemy.h
* @brief 課題9 : Enemyクラス定義
* @author IT11H308 ok230112 江藤大晴
* @date 2023/12/1
*/

#pragma once
class Enemy {
	std::string name;							/* 敵の名前 */
	int hitPoint;								/* 敵の体力 */
public:
	void setName(std::string na);/* 敵の名前の文字列を設定 */
	std::string getName();/* 敵の名前の文字列を取得 */
	void setHitPoint(int hp);/* 敵のHPの値を設定 */
	int getHitPoint();/* 敵のHPの値を取得 */
	void showData();/* ステータスを画面に表示 */
	void receiveDamage(int damage);/* ダメージを受け取る */
};
