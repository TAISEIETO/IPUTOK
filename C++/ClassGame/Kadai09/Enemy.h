/**
* @file Enemy.h
* @brief �ۑ�9 : Enemy�N���X��`
* @author IT11H308 ok230112 �]���吰
* @date 2023/12/1
*/

#pragma once
class Enemy {
	std::string name;							/* �G�̖��O */
	int hitPoint;								/* �G�̗̑� */
public:
	void setName(std::string na);/* �G�̖��O�̕������ݒ� */
	std::string getName();/* �G�̖��O�̕�������擾 */
	void setHitPoint(int hp);/* �G��HP�̒l��ݒ� */
	int getHitPoint();/* �G��HP�̒l���擾 */
	void showData();/* �X�e�[�^�X����ʂɕ\�� */
	void receiveDamage(int damage);/* �_���[�W���󂯎�� */
};
