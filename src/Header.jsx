import React from 'react';

const Header = () => {
	return (
		<header className="border-b bg-black border-slate-500 font-{Helvetica} font-bold px-5">
			<div className="mx-auto">
				<div className="flex justify-between h-20 items-center">
					<div className="w-60 bg-transparent">
						<img src={require('./LastLockLogoDark.svg').default} />
					</div>
				</div>
			</div>
		</header>
	);
};

export default Header;