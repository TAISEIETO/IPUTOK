/**
* @file Enemy.cpp
* @brief �ۑ�9 : Enemy�N���X�����o�[�֐�
* @author IT11H308 ok230112 �]���吰
* @date 2023/12/1
*/

#include <iostream>
using namespace std;

#include "Enemy.h"

/* ���O��ݒ� */
void Enemy::setName(string na) {
	name = na;
}

/* ���O���擾 */
string Enemy::getName() {
	return name;
}

/* HP��ݒ� */
void Enemy::setHitPoint(int hp) {
	hitPoint = hp;
}

/* HP���擾 */
int Enemy::getHitPoint() {
	return hitPoint;
}

/* ��Ԃ�\�� */
void Enemy::showData() {

	/* ���O�ƌ��݂�HP��\�� */
	cout << getName() << " HP=" << getHitPoint() << endl;
}

/* �_���[�W���󂯂� */
void Enemy::receiveDamage(int damage) {

	/* �_���[�W�̕\�� */
	cout << getName() << "��" << damage << "�̃_���[�W���󂯂��I" << endl;

	/* HP��damage������ */
	hitPoint -= damage;
	
	/* HP��0�ȉ��Ȃ�u�|�ꂽ�v*/
	if (hitPoint <= 0) {
		hitPoint = 0;
		cout << getName() << "�͓|�ꂽ�I" << endl;
	}
}