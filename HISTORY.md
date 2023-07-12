### Version 0.2.0

#### PRMSSnow

* [PRMSSnow](https://github.com/EC-USGS/pywatershed/commit/0cab6c4eeb1681006a141349359d797c0102d631): Ppt_to_pack and calin functional. Committed by James McCreight on 2022-04-28.

#### Compatibility

* [compatibility](https://github.com/EC-USGS/pywatershed/commit/e07ab484efd8d879d51a43f17cac990c226087e3): Convert 'bash' routines to Python equivalents to make it easier for Windows users to run the notebooks. Committed by Steve Westenbroek on 2022-12-07.
* [compatibility](https://github.com/EC-USGS/pywatershed/commit/849c17309d47d2a3b8349ab7a500db8ef806065c): Convert bash routines to python. Committed by Steve Westenbroek on 2022-12-08.

#### New features

* [feat](https://github.com/EC-USGS/pywatershed/commit/fc63c46e0e5ef34cc40a4c80cbb6799e499120af): Add Forcings and CSVFile classes (#14). Committed by jdhughes-usgs on 2022-02-22.
* [feat(control)](https://github.com/EC-USGS/pywatershed/commit/18b215c3d578912422f864bee26bddbca10261b6): Add functions to read data from the PRMS control file. Committed by Joseph D Hughes on 2022-03-17.
* [feat(control)](https://github.com/EC-USGS/pywatershed/commit/b60663b194eb535a02fc5d0322119d057ee96b6a): Add functions to read data from the PRMS control file. Committed by Joseph D Hughes on 2022-03-17.
* [feat(canopy)](https://github.com/EC-USGS/pywatershed/commit/f164e316f5bf85ffecead5c36223ae99d1572380): First canopy implementation for new framework. Committed by Langevin, Christian D on 2022-03-21.
* [feat(CsvFile)](https://github.com/EC-USGS/pywatershed/commit/39f394e6d2b8aa80fc1825bed52b971229dad05b): Add to_netcdf method (#42). Committed by jdhughes-usgs on 2022-03-24.
* [feat(canopy)](https://github.com/EC-USGS/pywatershed/commit/fee217e5ad4df1f2e65a75235af84d6d0432fc72): Adding more canopy calculations. Committed by Langevin, Christian D on 2022-03-24.
* [feat(gw)](https://github.com/EC-USGS/pywatershed/commit/f9226718ebb796fc79b8d5d873708058aa1097fb): Add groundwater component (#57). Committed by jdhughes-usgs on 2022-04-07.
* [feat(groundwater)](https://github.com/EC-USGS/pywatershed/commit/77813060571c2ab87928f85a47922fac96c3c98e): Add comparison of PRMS and pynhm groundwater results (#60). Committed by jdhughes-usgs on 2022-04-09.
* [feat(variableClass)](https://github.com/EC-USGS/pywatershed/commit/ab15b7d390c088491f28ceafc5fb3d5824ad6194): Add variable class (#61). Committed by jdhughes-usgs on 2022-04-13.
* [feat(channel)](https://github.com/EC-USGS/pywatershed/commit/9f0eb3ad16fb4650ad676785594d635286d69ca4): Add channel component (#80). Committed by jdhughes-usgs on 2022-05-18.
* [feat(canopy-runoff)](https://github.com/EC-USGS/pywatershed/commit/6a3b77ef5f30a933499e0b8fcfb3ba69eb243915): Wire up output from canopy as input to runoff (#86). Committed by langevin-usgs on 2022-05-26.
* [feat](https://github.com/EC-USGS/pywatershed/commit/ef3557957c509b6f33d97d6b226197f556dbd9f0): Json functions to instatiate PrmsParameters object from JSON and to save or load parameter dictionary as JSON. Committed by Mike Fienen on 2023-01-19.
* [feat](https://github.com/EC-USGS/pywatershed/commit/bf763b7ab21f9f0641155e111e1c081d5606f3c3): Automatic release to PyPi workflow. https://github.com/EC-USGS/pywatershed/pull/179. Committed by James McCreight on 2023-05-25.
* [feat](https://github.com/EC-USGS/pywatershed/commit/3b085e29db2ca8901a355187991e9d6df8084955): Individual process models run from separated parameters.. Committed by James McCreight on 2023-06-07.
* [feat](https://github.com/EC-USGS/pywatershed/commit/916ff976167ffb99dd1e7a45f2df8033b6233611): Model and Control from yaml files. Committed by James McCreight on 2023-06-21.
* [feat](https://github.com/EC-USGS/pywatershed/commit/998efbaf84320dda0f545bac3e8931ff211c5dee): Add pre-commit hook to strip output from jupyter notebooks. Committed by James McCreight on 2023-06-23.
* [feat](https://github.com/EC-USGS/pywatershed/commit/f53a5a96e934e6121bded7fe4a8bf452cd9e63d0): New notebook 01_multi-process_models. Committed by James McCreight on 2023-06-30.

#### Bug fixes

* [fix(pythonpath)](https://github.com/EC-USGS/pywatershed/commit/5dd3653b6ce5ed09a989a097a729339c20965e31): Remove hardcoded python path. Committed by Langevin, Christian D on 2022-03-22.
* [fix(canopy)](https://github.com/EC-USGS/pywatershed/commit/e3667fefc8b4395c88cfa83a9744a7f08e1f85e6): Minor canopy bug fix. Committed by Langevin, Christian D on 2022-03-30.
* [fix](https://github.com/EC-USGS/pywatershed/commit/9c92dabb9aeca2c0281bd9c6e3ac2742ff3e1526): Protect parameters as readonly np.arrays and MappingProxyTypes instead of dicts. Committed by James McCreight on 2023-05-26.

#### Reactor

* [reactor](https://github.com/EC-USGS/pywatershed/commit/c42808eb8f6e234f3999eb4fd0b777e77d008b2a): StorageUnit renamed to Process, before going to ConservativeProcess subclass Process is conservative. Committed by James McCreight on 2023-06-22.

#### Refactoring

* [refactor](https://github.com/EC-USGS/pywatershed/commit/f681e782b96920c17c73c52cb4579c82bce18f27): Reorganize reference data and prms source code. Committed by Joseph D Hughes on 2022-02-22.
* [refactor(parameters)](https://github.com/EC-USGS/pywatershed/commit/683d72c41ff8d6a522f375c309904913026da995): Refactor parameters. Committed by Joseph D Hughes on 2022-03-15.
* [refactor(parameters)](https://github.com/EC-USGS/pywatershed/commit/24830e4e94ad3464933d664730f2b5783db946e9): Refactor parameters. Committed by Joseph D Hughes on 2022-03-16.
* [refactor(parameters)](https://github.com/EC-USGS/pywatershed/commit/cee73d4d7d9de381a80e2060897a4fcc67587a59): Refactor parameters. Committed by Joseph D Hughes on 2022-03-16.
* [refactor(parameters)](https://github.com/EC-USGS/pywatershed/commit/0d94b3bd0edf93c65962780e4c24df5c5ac51168): Refactor parameters. Committed by Joseph D Hughes on 2022-03-16.
* [refactor(time)](https://github.com/EC-USGS/pywatershed/commit/61ded5473a87249e4cbe2d4bb8c303e70a37ebd7): Refactor time to use control file for initilization. Committed by Joseph D Hughes on 2022-03-16.
* [refactor(parameters)](https://github.com/EC-USGS/pywatershed/commit/ffec7288e8a9e48f79797cc8fef33bdce3ec9626): Finish generic PRMS file reader (#39). Committed by jdhughes-usgs on 2022-03-22.
* [refactor(makefile)](https://github.com/EC-USGS/pywatershed/commit/8bf2d32719b4c7991f5d1172c5d9d18b5e5f0de2): Refactor makefiles for multiple OSs and compilers (#41). Committed by jdhughes-usgs on 2022-03-23.
* [refactor(canopy)](https://github.com/EC-USGS/pywatershed/commit/f940cf09715bd9fd2edfbd925c108bae0ca8abae): Clean up canopy detritus. Committed by Langevin, Christian D on 2022-03-25.
* [refactor(canopy)](https://github.com/EC-USGS/pywatershed/commit/bf53d89497df7778cee71b1f64c175f7dc9f4919): Cleanup canopy detritus. Committed by langevin-usgs on 2022-03-25.
* [refactor(canopy)](https://github.com/EC-USGS/pywatershed/commit/c762673b4d1cadfeef0d1ac660e46c3ccc5e3c83): Work to match PRMS canopy calculations. Committed by Langevin, Christian D on 2022-03-29.
* [refactor(cnp)](https://github.com/EC-USGS/pywatershed/commit/7e813e869fb35fed453ed740943039b6a8aea65a): Get canopy working for UCB. Committed by Langevin, Christian D on 2022-04-01.
* [refactor(dirs)](https://github.com/EC-USGS/pywatershed/commit/611c8c68d6867fa5507705bb0b95aa425d259946): Refactor directory structure (#64). Committed by jdhughes-usgs on 2022-04-14.
* [refactor(PRMSGroundwater)](https://github.com/EC-USGS/pywatershed/commit/b195eeaf0949c4518b0af8d648c7c0e695ecbe43): Refactor groundwater to use controller (#65). Committed by jdhughes-usgs on 2022-04-14.
* [refactor(PRMSParameters)](https://github.com/EC-USGS/pywatershed/commit/de1d0d5ca5d27d9be13eb841c2480b8b87324fe3): Refactor PRMSParameters. (#66). Committed by jdhughes-usgs on 2022-04-14.
* [refactor(hru1)](https://github.com/EC-USGS/pywatershed/commit/4d443ed00ea4725003a54ca2405b38fc6afe38d8): Add modified version of hru_1 that has 1 segment (#79). Committed by jdhughes-usgs on 2022-05-04.
* [refactor(channel)](https://github.com/EC-USGS/pywatershed/commit/d424d07d327e4555b38288929a28347b8420d889): Refactor channel based on review comments (#83). Committed by jdhughes-usgs on 2022-05-18.
* [refactor](https://github.com/EC-USGS/pywatershed/commit/c82b98a528b33b6e81cf29a8234b6bf13c612e85): Refactor dependencies for standard installation. Committed by Joseph Hughes on 2023-05-08.
* [refactor](https://github.com/EC-USGS/pywatershed/commit/e01099277e8e42bd8a5b900eca3ad9170debf910): PRMSChannel init self._muskingum_mann from numpy, numba, fortran instead if during calculate. Committed by James McCreight on 2023-05-26.
* [refactor](https://github.com/EC-USGS/pywatershed/commit/c5a0ce10f3bf487d7cd508d7daac9494d92102ce): Separated PRMS parameters into discretizations for hrus and segments (dis_hru, dis_seg) and onto individual PRMS processes defined in pywatershed. Committed by James McCreight on 2023-06-14.
* [refactor](https://github.com/EC-USGS/pywatershed/commit/6be7331a8ec40835d7f0a4061c4b9cf5285b5e68): Control does not take/know parameters, pass dis and parameters to individual processes, individual process tests passing. Committed by James McCreight on 2023-06-15.
* [refactor](https://github.com/EC-USGS/pywatershed/commit/afbe91c296f3f78e211e7fa75e2c2dc81fd5375d): Model now takes a dictionary of [control, dis, process, order] while maintaining backwards compatability, very minor changes to api (passing list instead of unpacking list for arguments).. Committed by James McCreight on 2023-06-16.
* [refactor](https://github.com/EC-USGS/pywatershed/commit/cc44610c2d1c665336fe33074e4bd8d0908c2c0c): Calc_method clean up on channel and gw. Committed by James McCreight on 2023-06-22.
* [refactor](https://github.com/EC-USGS/pywatershed/commit/ff872fabfd406ddcf96653e7434034402c9e00ba): Calc_method clean up on canopy. Committed by James McCreight on 2023-06-22.
* [refactor](https://github.com/EC-USGS/pywatershed/commit/7310f2c7e8194bcfd5465f985dd411cebdd5b7f1): Calc_method clean up on prms snow. Committed by James McCreight on 2023-06-22.
* [refactor](https://github.com/EC-USGS/pywatershed/commit/cedd1e72f9a464e758e98698a514ea874dc63d15): Calc_method clean up on prms runoff and soilzone. Committed by James McCreight on 2023-06-22.
* [refactor](https://github.com/EC-USGS/pywatershed/commit/a78c75ab8fec90501dcb3a1d30fa63df48a23f63): Rename StorageUnit to ConservativeProcess class, subclass it from a Process class which does not have a budget. remove options from model except for input_dir, all other options go via control, new set_options on Process and ConservativeProcess. Committed by James McCreight on 2023-06-23.

#### Self-driving

* [self-driving](https://github.com/EC-USGS/pywatershed/commit/9848b905c85bdba11fc41f54f3e93ab3e65da01e): All but channel budget. Committed by James McCreight on 2023-05-09.
* [self-driving](https://github.com/EC-USGS/pywatershed/commit/33cdd89375640960ad7cd69796ddda16061b40fb): Channel variables to track inputs to channel and fix mass balance. Committed by James McCreight on 2023-05-09.

#### Soilzone

* [soilzone](https://github.com/EC-USGS/pywatershed/commit/25615a8fdf74af8f718bb10c3a68c34d80436386): Soil_moist and soil_rechr matching, others close, few off. Committed by James McCreight on 2022-05-16.

