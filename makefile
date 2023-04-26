all:
	echo "README\n"

push:
	git add -f .
	git commit -m "Update"
	git push origin gh-pages

TrueSkillThroughTime.py/.git:
	git submodule update --init TrueSkillThroughTime.py/

crear: TrueSkillThroughTime.py/.git
	make -C TrueSkillThroughTime.py/docs/

publicar: crear
	rsync -avrc --delete --exclude=.git --exclude=.gitmodules --exclude=makefile --exclude=index.html --exclude=TrueSkillThroughTime.py/ TrueSkillThroughTime.py/docs/build/ ./

